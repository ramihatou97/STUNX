"""
Text Processing Utilities for KOO Platform
Simple text processing functions without heavy dependencies
"""

import re
from typing import List, Set, Dict, Optional
from collections import Counter

def clean_text(text: str) -> str:
    """
    Clean and normalize text content

    Args:
        text: Raw text to clean

    Returns:
        Cleaned text
    """
    if not text:
        return ""

    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text.strip())

    # Remove HTML tags if present
    text = re.sub(r'<[^>]+>', '', text)

    # Remove special characters but keep medical notation
    text = re.sub(r'[^\w\s\-\.\,\;\:\!\?\(\)\[\]\/\%\+\=\<\>]', '', text)

    return text.strip()

def extract_keywords(text: str, min_length: int = 3, max_keywords: int = 20) -> List[str]:
    """
    Extract keywords from text using simple frequency analysis

    Args:
        text: Text to extract keywords from
        min_length: Minimum keyword length
        max_keywords: Maximum number of keywords to return

    Returns:
        List of extracted keywords
    """
    if not text:
        return []

    # Clean text and convert to lowercase
    clean = clean_text(text).lower()

    # Define stop words (medical-specific)
    stop_words = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'of', 'with', 'by', 'from', 'up', 'about', 'into', 'through', 'during',
        'before', 'after', 'above', 'below', 'between', 'among', 'this', 'that',
        'these', 'those', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
        'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
        'should', 'may', 'might', 'must', 'can', 'not', 'no', 'yes', 'all',
        'any', 'some', 'each', 'every', 'most', 'such', 'only', 'own', 'same',
        'so', 'than', 'too', 'very', 'just', 'now', 'here', 'there', 'when',
        'where', 'why', 'how', 'what', 'which', 'who', 'whom', 'whose'
    }

    # Extract words
    words = re.findall(r'\b[a-zA-Z]+\b', clean)

    # Filter words
    filtered_words = [
        word for word in words
        if len(word) >= min_length and word not in stop_words
    ]

    # Count frequencies
    word_counts = Counter(filtered_words)

    # Return most common words
    return [word for word, count in word_counts.most_common(max_keywords)]

def extract_medical_terms(text: str) -> List[str]:
    """
    Extract medical terms and anatomical references from text

    Args:
        text: Text to extract medical terms from

    Returns:
        List of medical terms found
    """
    if not text:
        return []

    # Medical term patterns
    medical_patterns = [
        # Anatomical terms
        r'\b(?:lobe|cortex|ventricle|hemisphere|sulcus|gyrus|fissure)\b',
        r'\b(?:frontal|parietal|temporal|occipital|limbic|cerebellar)\b',
        r'\b(?:anterior|posterior|superior|inferior|medial|lateral)\b',
        r'\b(?:spinal|cervical|thoracic|lumbar|sacral|coccygeal)\b',

        # Medical conditions
        r'\b(?:tumor|carcinoma|adenoma|meningioma|glioma|astrocytoma)\b',
        r'\b(?:aneurysm|stroke|hemorrhage|ischemia|edema|lesion)\b',
        r'\b(?:hydrocephalus|epilepsy|seizure|migraine|headache)\b',

        # Procedures
        r'\b(?:craniotomy|craniectomy|biopsy|resection|excision)\b',
        r'\b(?:surgery|surgical|operation|procedure|intervention)\b',
        r'\b(?:radiation|chemotherapy|immunotherapy|treatment)\b',

        # Measurements and values
        r'\b\d+(?:\.\d+)?\s*(?:mm|cm|ml|mg|units?)\b',
        r'\b(?:grade|stage|level|score)\s*[IVX0-9]+\b'
    ]

    found_terms = []
    text_lower = text.lower()

    for pattern in medical_patterns:
        matches = re.findall(pattern, text_lower, re.IGNORECASE)
        found_terms.extend(matches)

    # Remove duplicates while preserving order
    unique_terms = []
    seen = set()
    for term in found_terms:
        if term not in seen:
            unique_terms.append(term)
            seen.add(term)

    return unique_terms

def normalize_medical_term(term: str) -> str:
    """
    Normalize medical terms for consistent matching

    Args:
        term: Medical term to normalize

    Returns:
        Normalized term
    """
    if not term:
        return ""

    # Convert to lowercase and strip
    normalized = term.lower().strip()

    # Common medical abbreviations and expansions
    abbreviations = {
        'ca': 'carcinoma',
        'met': 'metastasis',
        'mri': 'magnetic resonance imaging',
        'ct': 'computed tomography',
        'csf': 'cerebrospinal fluid',
        'icp': 'intracranial pressure',
        'gcs': 'glasgow coma scale',
        'who': 'world health organization'
    }

    if normalized in abbreviations:
        return abbreviations[normalized]

    return normalized

def calculate_text_similarity(text1: str, text2: str) -> float:
    """
    Calculate simple text similarity using word overlap

    Args:
        text1: First text
        text2: Second text

    Returns:
        Similarity score between 0 and 1
    """
    if not text1 or not text2:
        return 0.0

    # Extract keywords from both texts
    keywords1 = set(extract_keywords(text1))
    keywords2 = set(extract_keywords(text2))

    if not keywords1 or not keywords2:
        return 0.0

    # Calculate Jaccard similarity
    intersection = len(keywords1.intersection(keywords2))
    union = len(keywords1.union(keywords2))

    return intersection / union if union > 0 else 0.0

def extract_citations(text: str) -> List[str]:
    """
    Extract citation-like patterns from text

    Args:
        text: Text to extract citations from

    Returns:
        List of potential citations
    """
    if not text:
        return []

    # Citation patterns
    citation_patterns = [
        # Journal citations (Author et al., Year)
        r'\b[A-Z][a-z]+\s+et\s+al\.\,?\s*\(?\d{4}\)?',

        # Direct citations (Smith, 2023)
        r'\b[A-Z][a-z]+\,?\s*\(?\d{4}\)?',

        # DOI patterns
        r'10\.\d+\/[^\s]+',

        # PubMed IDs
        r'PMID:\s*\d+',

        # Reference numbers [1], (1)
        r'[\[\(]\d+[\]\)]'
    ]

    citations = []
    for pattern in citation_patterns:
        matches = re.findall(pattern, text)
        citations.extend(matches)

    return list(set(citations))  # Remove duplicates

def summarize_text(text: str, max_sentences: int = 3) -> str:
    """
    Create a simple extractive summary of text

    Args:
        text: Text to summarize
        max_sentences: Maximum number of sentences in summary

    Returns:
        Text summary
    """
    if not text:
        return ""

    # Split into sentences
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]

    if len(sentences) <= max_sentences:
        return text

    # Simple scoring: sentences with more medical terms score higher
    scored_sentences = []
    for sentence in sentences:
        medical_terms = extract_medical_terms(sentence)
        score = len(medical_terms) + len(sentence.split()) * 0.1
        scored_sentences.append((score, sentence))

    # Sort by score and take top sentences
    scored_sentences.sort(reverse=True, key=lambda x: x[0])
    top_sentences = [s[1] for s in scored_sentences[:max_sentences]]

    return '. '.join(top_sentences) + '.'