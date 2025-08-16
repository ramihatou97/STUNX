import { Timestamp } from 'firebase/firestore';

/**
 * Represents a post-call assignment that crosses month boundaries
 * and needs to be handled by the next month's scheduler
 */
export interface CrossMonthPostCall {
    /**
     * ID of the resident who was on call
     */
    residentId: string;
    
    /**
     * Date for the post-call (will be in the next month)
     */
    date: Timestamp | Date;
    
    /**
     * ID of the original call assignment that generated this post-call
     */
    originatingCallId: string;
    
    /**
     * Date of the originating call
     */
    originatingCallDate: Timestamp;
}