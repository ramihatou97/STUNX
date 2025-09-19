import DOMPurify from 'isomorphic-dompurify';

export function sanitizeInput(input: any): any {
    if (input === null || input === undefined) {
        return input;
    }
    
    if (typeof input === 'string') {
        // Remove any HTML/script tags
        const cleaned = DOMPurify.sanitize(input, { 
            ALLOWED_TAGS: [],
            ALLOWED_ATTR: []
        });
        
        // Additional sanitization for SQL-like injections
        return cleaned
            .replace(/[;--]/g, '')
            .replace(/\/\*/g, '')
            .replace(/\*\//g, '')
            .trim();
    }
    
    if (Array.isArray(input)) {
        return input.map(sanitizeInput);
    }
    
    if (input instanceof Date || input.constructor?.name === 'Timestamp') {
        return input; // Don't sanitize dates/timestamps
    }
    
    if (typeof input === 'object') {
        const sanitized: any = {};
        for (const [key, value] of Object.entries(input)) {
            // Sanitize the key as well
            const sanitizedKey = sanitizeInput(key);
            if (typeof sanitizedKey === 'string' && sanitizedKey.length > 0) {
                sanitized[sanitizedKey] = sanitizeInput(value);
            }
        }
        return sanitized;
    }
    
    return input;
}

// Validate and sanitize email
export function sanitizeEmail(email: string): string {
    const sanitized = sanitizeInput(email);
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    
    if (!emailRegex.test(sanitized)) {
        throw new Error('Invalid email format');
    }
    
    return sanitized.toLowerCase();
}

// Validate and sanitize phone number
export function sanitizePhone(phone: string): string {
    // Remove all non-numeric characters
    const cleaned = phone.replace(/\D/g, '');
    
    // Check if it's a valid phone number length
    if (cleaned.length < 10 || cleaned.length > 15) {
        throw new Error('Invalid phone number');
    }
    
    return cleaned;
}

// Sanitize file names
export function sanitizeFileName(fileName: string): string {
    return fileName
        .replace(/[^a-zA-Z0-9.-]/g, '_')
        .replace(/_{2,}/g, '_')
        .toLowerCase();
}