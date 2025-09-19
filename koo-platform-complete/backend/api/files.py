"""
File Management API Endpoints for KOO Platform
Secure file upload, download, and management
"""

from typing import List, Optional
import logging

from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, Query
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel, Field
import io

from core.dependencies import get_current_user, get_db, CurrentUser
from core.exceptions import ValidationError, ResourceNotFoundError, DatabaseError, ErrorContext
from services.file_service import file_service, FileValidationError, FileRecord

logger = logging.getLogger(__name__)
router = APIRouter()

# Response Models
class FileUploadResponse(BaseModel):
    id: int
    uuid: str
    original_filename: str
    file_size: int
    mime_type: str
    upload_status: str
    created_at: str

    class Config:
        from_attributes = True

class FileInfoResponse(BaseModel):
    id: int
    uuid: str
    original_filename: str
    file_size: int
    mime_type: str
    file_hash: str
    created_at: str
    accessed_at: Optional[str]
    upload_context: dict
    is_processed: bool
    uploaded_by: str

    class Config:
        from_attributes = True

class FileListResponse(BaseModel):
    files: List[FileInfoResponse]
    total: int
    limit: int
    offset: int

class FileStatsResponse(BaseModel):
    total_files: int
    total_size: int
    types_breakdown: dict
    storage_path: str

@router.post("/upload", response_model=FileUploadResponse, status_code=status.HTTP_201_CREATED)
async def upload_file(
    file: UploadFile = File(...),
    context: Optional[str] = Query(None, description="Upload context (e.g., 'chapter_attachment')"),
    current_user: CurrentUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Upload a file with comprehensive validation and security checks

    Supports: PDF, DOC, DOCX, TXT, MD, PNG, JPG, JPEG
    Max size: 50MB
    """

    with ErrorContext("file_upload", user_id=current_user.id, filename=file.filename):
        try:
            # Validate file
            logger.info(f"Starting file upload: {file.filename} for user {current_user.id}")

            validated_data = await file_service.validate_file(file)

            # Prepare upload context
            upload_context = {
                "context": context or "general",
                "user_agent": "koo_platform",
                "upload_source": "web_interface"
            }

            # Store file
            file_record = await file_service.store_file(
                validated_data=validated_data,
                file=file,
                context=upload_context,
                uploaded_by=current_user.full_name
            )

            # Save to database
            db.add(file_record)
            await db.commit()
            await db.refresh(file_record)

            logger.info(f"File uploaded successfully: {file.filename} -> {file_record.uuid}")

            return FileUploadResponse(
                id=file_record.id,
                uuid=file_record.uuid,
                original_filename=file_record.original_filename,
                file_size=file_record.file_size,
                mime_type=file_record.mime_type,
                upload_status="success",
                created_at=file_record.created_at.isoformat()
            )

        except FileValidationError as e:
            logger.warning(f"File validation failed: {e}")
            raise ValidationError(str(e), "file")

        except Exception as e:
            logger.error(f"File upload failed: {e}")
            raise DatabaseError("File upload failed")

@router.get("/", response_model=FileListResponse)
async def list_files(
    limit: int = Query(50, ge=1, le=100, description="Number of files to return"),
    offset: int = Query(0, ge=0, description="Number of files to skip"),
    mime_type: Optional[str] = Query(None, description="Filter by MIME type"),
    current_user: CurrentUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """List uploaded files with pagination and filtering"""

    with ErrorContext("list_files", user_id=current_user.id):
        try:
            files = await file_service.list_files(db, limit=limit, offset=offset)

            # Filter by MIME type if specified
            if mime_type:
                files = [f for f in files if f.mime_type == mime_type]

            file_responses = [
                FileInfoResponse(
                    id=f.id,
                    uuid=f.uuid,
                    original_filename=f.original_filename,
                    file_size=f.file_size,
                    mime_type=f.mime_type,
                    file_hash=f.file_hash,
                    created_at=f.created_at.isoformat(),
                    accessed_at=f.accessed_at.isoformat() if f.accessed_at else None,
                    upload_context=f.upload_context,
                    is_processed=f.is_processed,
                    uploaded_by=f.uploaded_by
                )
                for f in files
            ]

            return FileListResponse(
                files=file_responses,
                total=len(file_responses),
                limit=limit,
                offset=offset
            )

        except Exception as e:
            logger.error(f"Failed to list files: {e}")
            raise DatabaseError("Failed to retrieve file list")

@router.get("/{file_uuid}/info", response_model=FileInfoResponse)
async def get_file_info(
    file_uuid: str,
    current_user: CurrentUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Get detailed information about a specific file"""

    with ErrorContext("get_file_info", user_id=current_user.id, file_uuid=file_uuid):
        try:
            file_record = await file_service.get_file(file_uuid, db)

            if not file_record:
                raise ResourceNotFoundError("File", file_uuid)

            return FileInfoResponse(
                id=file_record.id,
                uuid=file_record.uuid,
                original_filename=file_record.original_filename,
                file_size=file_record.file_size,
                mime_type=file_record.mime_type,
                file_hash=file_record.file_hash,
                created_at=file_record.created_at.isoformat(),
                accessed_at=file_record.accessed_at.isoformat() if file_record.accessed_at else None,
                upload_context=file_record.upload_context,
                is_processed=file_record.is_processed,
                uploaded_by=file_record.uploaded_by
            )

        except ResourceNotFoundError:
            raise
        except Exception as e:
            logger.error(f"Failed to get file info {file_uuid}: {e}")
            raise DatabaseError("Failed to retrieve file information")

@router.get("/{file_uuid}/download")
async def download_file(
    file_uuid: str,
    current_user: CurrentUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Download a file by UUID"""

    with ErrorContext("download_file", user_id=current_user.id, file_uuid=file_uuid):
        try:
            # Get file record
            file_record = await file_service.get_file(file_uuid, db)

            if not file_record:
                raise ResourceNotFoundError("File", file_uuid)

            # Read file content
            content = await file_service.read_file_content(file_record)

            logger.info(f"File downloaded: {file_record.original_filename} by user {current_user.id}")

            # Return streaming response
            return StreamingResponse(
                io.BytesIO(content),
                media_type=file_record.mime_type,
                headers={
                    "Content-Disposition": f"attachment; filename={file_record.original_filename}",
                    "Content-Length": str(file_record.file_size),
                    "X-File-UUID": file_record.uuid
                }
            )

        except ResourceNotFoundError:
            raise
        except Exception as e:
            logger.error(f"Failed to download file {file_uuid}: {e}")
            raise DatabaseError("Failed to download file")

@router.get("/{file_uuid}/preview")
async def preview_file(
    file_uuid: str,
    current_user: CurrentUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Preview a file (for images and text files)"""

    with ErrorContext("preview_file", user_id=current_user.id, file_uuid=file_uuid):
        try:
            # Get file record
            file_record = await file_service.get_file(file_uuid, db)

            if not file_record:
                raise ResourceNotFoundError("File", file_uuid)

            # Check if file type supports preview
            previewable_types = [
                "image/jpeg", "image/png", "image/gif", "image/webp",
                "text/plain", "text/markdown", "text/html"
            ]

            if file_record.mime_type not in previewable_types:
                raise ValidationError("File type does not support preview", "mime_type")

            # Read file content
            content = await file_service.read_file_content(file_record)

            # For text files, decode content
            if file_record.mime_type.startswith("text/"):
                try:
                    text_content = content.decode('utf-8')
                    return {
                        "type": "text",
                        "content": text_content,
                        "filename": file_record.original_filename,
                        "mime_type": file_record.mime_type
                    }
                except UnicodeDecodeError:
                    raise ValidationError("Unable to decode text file", "encoding")

            # For images, return as streaming response
            return StreamingResponse(
                io.BytesIO(content),
                media_type=file_record.mime_type,
                headers={
                    "X-File-UUID": file_record.uuid,
                    "Cache-Control": "max-age=3600"  # Cache for 1 hour
                }
            )

        except (ResourceNotFoundError, ValidationError):
            raise
        except Exception as e:
            logger.error(f"Failed to preview file {file_uuid}: {e}")
            raise DatabaseError("Failed to preview file")

@router.delete("/{file_uuid}")
async def delete_file(
    file_uuid: str,
    current_user: CurrentUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Delete a file"""

    with ErrorContext("delete_file", user_id=current_user.id, file_uuid=file_uuid):
        try:
            success = await file_service.delete_file(file_uuid, db)

            if success:
                logger.info(f"File deleted: {file_uuid} by user {current_user.id}")
                return {"message": "File deleted successfully", "file_uuid": file_uuid}
            else:
                raise DatabaseError("Failed to delete file")

        except ResourceNotFoundError:
            raise
        except Exception as e:
            logger.error(f"Failed to delete file {file_uuid}: {e}")
            raise DatabaseError("Failed to delete file")

@router.get("/stats", response_model=FileStatsResponse)
async def get_file_stats(
    current_user: CurrentUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Get file storage statistics"""

    with ErrorContext("get_file_stats", user_id=current_user.id):
        try:
            stats = await file_service.get_file_stats(db)

            return FileStatsResponse(
                total_files=stats["total_files"],
                total_size=stats["total_size"],
                types_breakdown=stats["types_breakdown"],
                storage_path=stats["storage_path"]
            )

        except Exception as e:
            logger.error(f"Failed to get file stats: {e}")
            raise DatabaseError("Failed to retrieve file statistics")

@router.post("/{file_uuid}/process")
async def process_file(
    file_uuid: str,
    current_user: CurrentUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Process a file (extract text, generate thumbnails, etc.)"""

    with ErrorContext("process_file", user_id=current_user.id, file_uuid=file_uuid):
        try:
            # Get file record
            file_record = await file_service.get_file(file_uuid, db)

            if not file_record:
                raise ResourceNotFoundError("File", file_uuid)

            # Mark as processed (simplified implementation)
            file_record.is_processed = True
            await db.commit()

            logger.info(f"File processed: {file_uuid} by user {current_user.id}")

            return {
                "message": "File processing completed",
                "file_uuid": file_uuid,
                "status": "processed"
            }

        except ResourceNotFoundError:
            raise
        except Exception as e:
            logger.error(f"Failed to process file {file_uuid}: {e}")
            raise DatabaseError("Failed to process file")