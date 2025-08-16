import { useState, useCallback, useRef } from 'react';
import { requestManager } from '../services/request-manager';
import toast from 'react-hot-toast';

interface UsePreventDuplicatesOptions {
  minInterval?: number;
  showToast?: boolean;
  customKey?: string;
}

export function usePreventDuplicates<T extends (...args: any[]) => Promise<any>>(
  asyncFunction: T,
  options: UsePreventDuplicatesOptions = {}
): [T, boolean, Error | null] {
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<Error | null>(null);
  const lastCallTime = useRef<number>(0);
  
  const wrappedFunction = useCallback(
    async (...args: Parameters<T>) => {
      const now = Date.now();
      const minInterval = options.minInterval || 2000;
      
      // Check minimum interval
      if (now - lastCallTime.current < minInterval) {
        const waitTime = Math.ceil((minInterval - (now - lastCallTime.current)) / 1000);
        const errorMsg = `Please wait ${waitTime} seconds before retrying`;
        
        if (options.showToast !== false) {
          toast.error(errorMsg, { duration: 3000 });
        }
        
        const err = new Error(errorMsg);
        setError(err);
        throw err;
      }
      
      // Check if already loading
      if (isLoading) {
        const errorMsg = 'Request already in progress';
        
        if (options.showToast !== false) {
          toast.warning(errorMsg);
        }
        
        const err = new Error(errorMsg);
        setError(err);
        throw err;
      }
      
      setIsLoading(true);
      setError(null);
      lastCallTime.current = now;
      
      try {
        const result = await asyncFunction(...args);
        return result;
      } catch (err: any) {
        setError(err);
        throw err;
      } finally {
        setIsLoading(false);
      }
    },
    [asyncFunction, isLoading, options.minInterval, options.showToast]
  ) as T;
  
  return [wrappedFunction, isLoading, error];
}