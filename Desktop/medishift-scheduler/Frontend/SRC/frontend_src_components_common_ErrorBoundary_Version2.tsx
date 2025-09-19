import React, { Component, ErrorInfo, ReactNode } from 'react';
import * as Sentry from '@sentry/react';

interface Props {
    children: ReactNode;
    fallback?: ReactNode;
    showDetails?: boolean;
    onError?: (error: Error, errorInfo: ErrorInfo) => void;
}

interface State {
    hasError: boolean;
    error?: Error;
    errorInfo?: ErrorInfo;
    errorId?: string;
}

export class ErrorBoundary extends Component<Props, State> {
    state: State = { 
        hasError: false 
    };
    
    static getDerivedStateFromError(error: Error): State {
        const errorId = Math.random().toString(36).substring(7);
        return { 
            hasError: true, 
            error,
            errorId 
        };
    }
    
    componentDidCatch(error: Error, errorInfo: ErrorInfo) {
        console.error('Error caught by boundary:', error, errorInfo);
        
        // Log to Sentry in production
        if (process.env.NODE_ENV === 'production') {
            Sentry.withScope((scope) => {
                scope.setTag('errorBoundary', true);
                scope.setContext('errorInfo', errorInfo);
                scope.setContext('errorId', this.state.errorId);
                Sentry.captureException(error);
            });
        }
        
        // Call custom error handler if provided
        if (this.props.onError) {
            this.props.onError(error, errorInfo);
        }
        
        // Update state with error info
        this.setState({ errorInfo });
    }
    
    handleReset = () => {
        this.setState({ 
            hasError: false, 
            error: undefined, 
            errorInfo: undefined,
            errorId: undefined 
        });
    };
    
    render() {
        if (this.state.hasError) {
            if (this.props.fallback) {
                return this.props.fallback;
            }
            
            return (
                <div className="min-h-screen flex items-center justify-center bg-gray-50">
                    <div className="max-w-md w-full bg-white shadow-lg rounded-lg p-6">
                        <div className="flex items-center mb-4">
                            <div className="flex-shrink-0">
                                <svg 
                                    className="h-12 w-12 text-red-500" 
                                    fill="none" 
                                    viewBox="0 0 24 24" 
                                    stroke="currentColor"
                                >
                                    <path 
                                        strokeLinecap="round" 
                                        strokeLinejoin="round" 
                                        strokeWidth={2} 
                                        d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" 
                                    />
                                </svg>
                            </div>
                            <div className="ml-3">
                                <h3 className="text-lg font-medium text-gray-900">
                                    Something went wrong
                                </h3>
                                <p className="text-sm text-gray-500">
                                    Error ID: {this.state.errorId}
                                </p>
                            </div>
                        </div>
                        
                        <div className="mt-4">
                            <p className="text-sm text-gray-600">
                                {this.state.error?.message || 'An unexpected error occurred'}
                            </p>
                            
                            {this.props.showDetails && this.state.errorInfo && (
                                <details className="mt-4">
                                    <summary className="cursor-pointer text-sm text-blue-600 hover:text-blue-800">
                                        Show technical details
                                    </summary>
                                    <pre className="mt-2 text-xs bg-gray-100 p-2 rounded overflow-auto">
                                        {this.state.errorInfo.componentStack}
                                    </pre>
                                </details>
                            )}
                        </div>
                        
                        <div className="mt-6 flex space-x-3">
                            <button
                                onClick={this.handleReset}
                                className="flex-1 bg-blue-600 text-white px-4 py-2 rounded-md hover:bg-blue-700 transition-colors"
                            >
                                Try Again
                            </button>
                            <button
                                onClick={() => window.location.href = '/'}
                                className="flex-1 bg-gray-200 text-gray-800 px-4 py-2 rounded-md hover:bg-gray-300 transition-colors"
                            >
                                Go Home
                            </button>
                        </div>
                        
                        {process.env.NODE_ENV === 'production' && (
                            <p className="mt-4 text-xs text-gray-500 text-center">
                                This error has been automatically reported to our team.
                            </p>
                        )}
                    </div>
                </div>
            );
        }
        
        return this.props.children;
    }
}

// HOC for easier use with functional components
export function withErrorBoundary<P extends object>(
    Component: React.ComponentType<P>,
    errorBoundaryProps?: Omit<Props, 'children'>
) {
    const WrappedComponent = (props: P) => (
        <ErrorBoundary {...errorBoundaryProps}>
            <Component {...props} />
        </ErrorBoundary>
    );
    
    WrappedComponent.displayName = `withErrorBoundary(${Component.displayName || Component.name})`;
    
    return WrappedComponent;
}

// React Error Boundary Hook (for functional components)
export function useErrorHandler() {
    return (error: Error, errorInfo?: { componentStack: string }) => {
        console.error('Error caught by hook:', error, errorInfo);
        
        if (process.env.NODE_ENV === 'production') {
            Sentry.withScope((scope) => {
                scope.setTag('errorBoundary', false);
                scope.setTag('errorHook', true);
                if (errorInfo) {
                    scope.setContext('errorInfo', errorInfo);
                }
                Sentry.captureException(error);
            });
        }
    };
}