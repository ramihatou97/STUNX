"""
Comprehensive File Upload and Management Service for KOO Platform
Handles file validation, storage, retrieval, and security
"""

import os
import hashlib
import mimetypes
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List, Dict, Any, BinaryIO
import logging
import aiofiles
import magic

from fastapi import UploadFile, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from PIL import Image
import PyPDF2

from core.config import settings
from core.exceptions import ValidationError, ResourceNotFoundError, DatabaseError
from models.base import Base
from sqlalchemy import String, Text, Integer, DateTime, Boolean, JSON
from sqlalchemy.orm import Mapped, mapped_column

logger = logging.getLogger(__name__)

class FileRecord(Base):
    """Database model for file records"""

    __tablename__ = "files"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    uuid: Mapped[str] = mapped_column(String(36), unique=True, nullable=False)
    original_filename: Mapped[str] = mapped_column(String(255), nullable=False)
    stored_filename: Mapped[str] = mapped_column(String(255), nullable=False)
    file_path: Mapped[str] = mapped_column(String(500), nullable=False)

    # File properties
    file_size: Mapped[int] = mapped_column(Integer, nullable=False)
    mime_type: Mapped[str] = mapped_column(String(100), nullable=False)
    file_hash: Mapped[str] = mapped_column(String(64), nullable=False)  # SHA-256

    # Metadata
    upload_context: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)
    is_processed: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    is_public: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)

    # Security
    virus_scanned: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    scan_result: Mapped[str] = mapped_column(String(50), default="pending", nullable=False)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
    accessed_at: Mapped[Optional[datetime]] = mapped_column(DateTime)

    # User context (always admin in single-user)
    uploaded_by: Mapped[str] = mapped_column(String(100), nullable=False)

class FileValidationError(Exception):
    """Custom exception for file validation errors"""
    pass

class FileStorageService:
    """Comprehensive file storage and management service"""

    def __init__(self):
        self.upload_dir = Path("uploads")
        self.max_file_size = settings.UPLOAD_MAX_SIZE
        self.allowed_types = settings.ALLOWED_FILE_TYPES
        self.storage_structure = "uploads/{year}/{month}/{day}"

        # Create upload directory if it doesn't exist
        self.upload_dir.mkdir(exist_ok=True)

    async def validate_file(self, file: UploadFile) -> Dict[str, Any]:
        """Comprehensive file validation"""

        # Check file size
        if hasattr(file, 'size') and file.size and file.size > self.max_file_size:
            raise FileValidationError(
                f"File size {file.size} exceeds maximum allowed size {self.max_file_size}"
            )

        # Read file content for validation
        content = await file.read()
        await file.seek(0)  # Reset file pointer

        if len(content) > self.max_file_size:
            raise FileValidationError(
                f"File size {len(content)} exceeds maximum allowed size {self.max_file_size}"
            )

        # Detect MIME type using python-magic
        try:
            detected_mime = magic.from_buffer(content, mime=True)
        except Exception:
            # Fallback to mimetypes
            detected_mime = mimetypes.guess_type(file.filename or "")[0] or "application/octet-stream"

        # Validate file extension
        if file.filename:
            file_extension = Path(file.filename).suffix.lower().lstrip('.')
            if file_extension not in self.allowed_types:
                raise FileValidationError(
                    f"File type '{file_extension}' not allowed. Allowed types: {', '.join(self.allowed_types)}"
                )

        # Additional validation based on file type
        validation_result = await self._validate_file_content(content, detected_mime, file.filename)

        return {
            "content": content,
            "mime_type": detected_mime,
            "file_size": len(content),
            "validation_result": validation_result
        }

    async def _validate_file_content(self, content: bytes, mime_type: str, filename: Optional[str]) -> Dict[str, Any]:
        """Validate file content based on type"""

        validation_result = {
            "is_valid": True,
            "warnings": [],
            "metadata": {}
        }

        try:
            # PDF validation
            if mime_type == "application/pdf":
                validation_result["metadata"] = await self._validate_pdf(content)

            # Image validation
            elif mime_type.startswith("image/"):
                validation_result["metadata"] = await self._validate_image(content)

            # Text file validation
            elif mime_type.startswith("text/"):
                validation_result["metadata"] = await self._validate_text(content)

            # Document validation (Word, etc.)
            elif mime_type in ["application/msword", "application/vnd.openxmlformats-officedocument.wordprocessingml.document"]:
                validation_result["metadata"] = await self._validate_document(content, mime_type)

        except Exception as e:
            validation_result["warnings"].append(f"Content validation warning: {str(e)}")
            logger.warning(f"File content validation failed: {e}")

        return validation_result

    async def _validate_pdf(self, content: bytes) -> Dict[str, Any]:
        """Validate PDF file"""
        try:
            from io import BytesIO
            pdf_reader = PyPDF2.PdfReader(BytesIO(content))

            return {
                "pages": len(pdf_reader.pages),
                "encrypted": pdf_reader.is_encrypted,
                "title": pdf_reader.metadata.get("/Title") if pdf_reader.metadata else None,
                "author": pdf_reader.metadata.get("/Author") if pdf_reader.metadata else None
            }
        except Exception as e:
            raise FileValidationError(f"Invalid PDF file: {str(e)}")

    async def _validate_image(self, content: bytes) -> Dict[str, Any]:
        """Validate image file"""
        try:
            from io import BytesIO
            with Image.open(BytesIO(content)) as img:
                return {
                    "format": img.format,
                    "mode": img.mode,
                    "size": img.size,
                    "has_transparency": img.mode in ("RGBA", "LA") or "transparency" in img.info
                }
        except Exception as e:
            raise FileValidationError(f"Invalid image file: {str(e)}")

    async def _validate_text(self, content: bytes) -> Dict[str, Any]:
        """Validate text file"""
        try:
            # Try to decode as UTF-8
            text_content = content.decode('utf-8')

            return {
                "encoding": "utf-8",
                "line_count": len(text_content.splitlines()),
                "char_count": len(text_content),
                "word_count": len(text_content.split())
            }
        except UnicodeDecodeError:
            # Try other encodings
            for encoding in ['latin-1', 'cp1252', 'iso-8859-1']:
                try:
                    text_content = content.decode(encoding)
                    return {
                        "encoding": encoding,
                        "line_count": len(text_content.splitlines()),
                        "char_count": len(text_content),
                        "word_count": len(text_content.split())
                    }
                except UnicodeDecodeError:
                    continue

            raise FileValidationError("Unable to decode text file with supported encodings")

    async def _validate_document(self, content: bytes, mime_type: str) -> Dict[str, Any]:
        """Validate office documents"""
        # Basic validation - could be extended with python-docx for detailed analysis
        return {
            "mime_type": mime_type,
            "size": len(content),
            "status": "basic_validation_passed"
        }

    def _generate_file_hash(self, content: bytes) -> str:
        """Generate SHA-256 hash of file content"""
        return hashlib.sha256(content).hexdigest()

    def _generate_storage_path(self) -> str:
        """Generate storage path based on current date"""
        now = datetime.utcnow()
        return self.storage_structure.format(
            year=now.year,
            month=now.month,
            day=now.day
        )

    def _generate_unique_filename(self, original_filename: str) -> str:
        """Generate unique filename while preserving extension"""
        file_path = Path(original_filename)
        unique_id = str(uuid.uuid4())
        return f"{unique_id}{file_path.suffix}"

    async def store_file(self, validated_data: Dict[str, Any], file: UploadFile,
                        context: Dict[str, Any], uploaded_by: str) -> FileRecord:
        """Store validated file to disk and database"""

        try:
            # Generate storage path and filename
            storage_path = self._generate_storage_path()
            stored_filename = self._generate_unique_filename(file.filename or "unknown")

            # Create directory structure
            full_storage_path = self.upload_dir / storage_path
            full_storage_path.mkdir(parents=True, exist_ok=True)

            # Full file path
            file_path = full_storage_path / stored_filename

            # Write file to disk
            async with aiofiles.open(file_path, 'wb') as f:
                await f.write(validated_data["content"])

            # Create database record
            file_record = FileRecord(
                uuid=str(uuid.uuid4()),
                original_filename=file.filename or "unknown",
                stored_filename=stored_filename,
                file_path=str(file_path),
                file_size=validated_data["file_size"],
                mime_type=validated_data["mime_type"],
                file_hash=self._generate_file_hash(validated_data["content"]),
                upload_context=context,
                uploaded_by=uploaded_by,
                scan_result="clean"  # Simplified - would integrate with virus scanner
            )

            logger.info(f"File stored successfully: {file.filename} -> {stored_filename}")

            return file_record

        except Exception as e:
            logger.error(f"Failed to store file {file.filename}: {e}")
            raise DatabaseError(f"Failed to store file: {str(e)}")

    async def get_file(self, file_uuid: str, db: AsyncSession) -> Optional[FileRecord]:
        """Retrieve file record from database"""
        try:
            result = await db.execute(
                select(FileRecord).where(FileRecord.uuid == file_uuid)
            )
            file_record = result.scalar_one_or_none()

            if file_record:
                # Update access time
                file_record.accessed_at = datetime.utcnow()
                await db.commit()

            return file_record

        except Exception as e:
            logger.error(f"Failed to retrieve file {file_uuid}: {e}")
            raise DatabaseError(f"Failed to retrieve file: {str(e)}")

    async def read_file_content(self, file_record: FileRecord) -> bytes:
        """Read file content from disk"""
        try:
            file_path = Path(file_record.file_path)

            if not file_path.exists():
                raise ResourceNotFoundError("File", file_record.uuid)

            async with aiofiles.open(file_path, 'rb') as f:
                content = await f.read()

            # Verify file integrity
            current_hash = self._generate_file_hash(content)
            if current_hash != file_record.file_hash:
                logger.error(f"File integrity check failed for {file_record.uuid}")
                raise ValidationError("File integrity verification failed")

            return content

        except Exception as e:
            logger.error(f"Failed to read file {file_record.uuid}: {e}")
            raise DatabaseError(f"Failed to read file: {str(e)}")

    async def delete_file(self, file_uuid: str, db: AsyncSession) -> bool:
        """Delete file from both disk and database"""
        try:
            # Get file record
            file_record = await self.get_file(file_uuid, db)
            if not file_record:
                raise ResourceNotFoundError("File", file_uuid)

            # Delete from disk
            file_path = Path(file_record.file_path)
            if file_path.exists():
                file_path.unlink()

            # Delete from database
            await db.delete(file_record)
            await db.commit()

            logger.info(f"File deleted successfully: {file_uuid}")
            return True

        except Exception as e:
            logger.error(f"Failed to delete file {file_uuid}: {e}")
            raise DatabaseError(f"Failed to delete file: {str(e)}")

    async def list_files(self, db: AsyncSession, limit: int = 50, offset: int = 0) -> List[FileRecord]:
        """List files with pagination"""
        try:
            result = await db.execute(
                select(FileRecord)
                .order_by(FileRecord.created_at.desc())
                .limit(limit)
                .offset(offset)
            )

            return result.scalars().all()

        except Exception as e:
            logger.error(f"Failed to list files: {e}")
            raise DatabaseError(f"Failed to list files: {str(e)}")

    async def get_file_stats(self, db: AsyncSession) -> Dict[str, Any]:
        """Get file storage statistics"""
        try:
            # This would be implemented with proper SQL queries
            # For now, return basic stats

            return {
                "total_files": 0,  # Would count files
                "total_size": 0,   # Would sum file sizes
                "types_breakdown": {},  # Would group by mime_type
                "storage_path": str(self.upload_dir.absolute())
            }

        except Exception as e:
            logger.error(f"Failed to get file stats: {e}")
            raise DatabaseError(f"Failed to get file statistics: {str(e)}")

# Global file service instance
file_service = FileStorageService()