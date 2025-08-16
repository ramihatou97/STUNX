import React, { createContext, useContext, useState, useCallback, ReactNode } from 'react';

interface LoadingContextType {
    isLoading: boolean;
    loadingMessage: string;
    loadingProgress: number;
    setLoading: (loading: boolean, message?: string) => void;
    setProgress: (progress: number) => void;
    withLoading: <T>(promise: Promise<T>, message?: string) => Promise<T>;
}

const LoadingContext = createContext<LoadingContextType | undefined>(undefined);

export const useLoading = () => {
    const context = useContext(LoadingContext);
    if (!context) {
        throw new Error('useLoading must be used within LoadingProvider');
    }
    return context;
};

interface LoadingProviderProps {
    children: ReactNode;
}

export const LoadingProvider: React.FC<LoadingProviderProps> = ({ children }) => {
    const [isLoading, setIsLoading] = useState(false);
    const [loadingMessage, setLoadingMessage] = useState('');
    const [loadingProgress, setLoadingProgress] = useState(0);
    
    const setLoading = useCallback((loading: boolean, message: string = '') => {
        setIsLoading(loading);
        setLoadingMessage(message);
        if (!loading) {
            setLoadingProgress(0);
        }
    }, []);
    
    const setProgress = useCallback((progress: number) => {
        setLoadingProgress(Math.min(100, Math.max(0, progress)));
    }, []);
    
    const withLoading = useCallback(async <T,>(
        promise: Promise<T>, 
        message: string = 'Loading