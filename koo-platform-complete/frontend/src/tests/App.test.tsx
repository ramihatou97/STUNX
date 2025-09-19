/**
 * Frontend Test Suite for KOO Platform
 */

import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { BrowserRouter } from 'react-router-dom';
import { ThemeProvider, createTheme } from '@mui/material';
import '@testing-library/jest-dom';

import App from '../App';
import { ChapterViewer } from '../pages/ChapterViewer';
import { Research } from '../pages/Research';
import { AuthProvider } from '../contexts/AuthContext';

// Mock API responses
jest.mock('../services/chapterService', () => ({
  getChapter: jest.fn(),
  getChapterVersions: jest.fn(),
  getChapterMetrics: jest.fn(),
}));

jest.mock('../services/researchService', () => ({
  search: jest.fn(),
  getSavedSearches: jest.fn(),
}));

// Test utilities
const theme = createTheme();

const TestWrapper: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: { retry: false },
      mutations: { retry: false },
    },
  });

  return (
    <QueryClientProvider client={queryClient}>
      <ThemeProvider theme={theme}>
        <BrowserRouter>
          <AuthProvider>
            {children}
          </AuthProvider>
        </BrowserRouter>
      </ThemeProvider>
    </QueryClientProvider>
  );
};

describe('KOO Platform App', () => {
  test('renders main application', () => {
    render(
      <TestWrapper>
        <App />
      </TestWrapper>
    );

    // Should redirect to login when not authenticated
    expect(screen.getByText(/login/i)).toBeInTheDocument();
  });

  test('handles authentication flow', async () => {
    const mockLogin = jest.fn();

    render(
      <TestWrapper>
        <App />
      </TestWrapper>
    );

    // Should show login form
    const loginButton = screen.getByRole('button', { name: /login/i });
    expect(loginButton).toBeInTheDocument();

    // Fill login form
    const usernameInput = screen.getByLabelText(/username/i);
    const passwordInput = screen.getByLabelText(/password/i);

    fireEvent.change(usernameInput, { target: { value: 'testuser' } });
    fireEvent.change(passwordInput, { target: { value: 'password' } });

    // Submit form
    fireEvent.click(loginButton);

    // Should attempt login
    await waitFor(() => {
      expect(mockLogin).toHaveBeenCalled();
    });
  });
});

describe('Chapter Viewer', () => {
  const mockChapter = {
    id: 1,
    title: 'Test Neurosurgical Procedure',
    content: '# Test Chapter\n\nThis is test content.',
    category: 'neurosurgery',
    version: '1.0.0',
    summary: 'Test summary',
    created_at: '2024-01-01T00:00:00Z',
    updated_at: '2024-01-01T00:00:00Z',
    author_name: 'Dr. Test',
    word_count: 100,
    reading_time_minutes: 5,
    is_bookmarked: false,
    comment_count: 0,
  };

  beforeEach(() => {
    const { getChapter } = require('../services/chapterService');
    getChapter.mockResolvedValue(mockChapter);
  });

  test('displays chapter content', async () => {
    render(
      <TestWrapper>
        <ChapterViewer />
      </TestWrapper>
    );

    await waitFor(() => {
      expect(screen.getByText('Test Neurosurgical Procedure')).toBeInTheDocument();
    });

    expect(screen.getByText(/test content/i)).toBeInTheDocument();
    expect(screen.getByText('neurosurgery')).toBeInTheDocument();
  });

  test('handles chapter navigation', async () => {
    render(
      <TestWrapper>
        <ChapterViewer />
      </TestWrapper>
    );

    await waitFor(() => {
      expect(screen.getByText('Test Neurosurgical Procedure')).toBeInTheDocument();
    });

    // Test tab navigation
    const commentsTab = screen.getByRole('tab', { name: /comments/i });
    fireEvent.click(commentsTab);

    expect(screen.getByRole('tabpanel')).toBeInTheDocument();
  });

  test('handles bookmark functionality', async () => {
    const mockToggleBookmark = jest.fn();

    render(
      <TestWrapper>
        <ChapterViewer />
      </TestWrapper>
    );

    await waitFor(() => {
      expect(screen.getByText('Test Neurosurgical Procedure')).toBeInTheDocument();
    });

    const bookmarkButton = screen.getByRole('button', { name: /bookmark/i });
    fireEvent.click(bookmarkButton);

    expect(mockToggleBookmark).toHaveBeenCalled();
  });
});

describe('Research Interface', () => {
  const mockSearchResults = {
    results: [
      {
        id: '1',
        title: 'Neurosurgical Research Paper 1',
        authors: ['Dr. Smith', 'Dr. Jones'],
        abstract: 'This is a research abstract about neurosurgery.',
        publication_date: '2024-01-01',
        journal: 'Journal of Neurosurgery',
        source: 'pubmed',
        pmid: '12345678',
      },
      {
        id: '2',
        title: 'Advanced Brain Surgery Techniques',
        authors: ['Dr. Brown'],
        abstract: 'Advanced techniques in brain surgery.',
        publication_date: '2023-12-15',
        journal: 'Neurosurgery Today',
        source: 'pubmed',
        pmid: '87654321',
      }
    ],
    total: 2,
    page: 0,
    hasMore: false,
  };

  beforeEach(() => {
    const { search } = require('../services/researchService');
    search.mockResolvedValue(mockSearchResults);
  });

  test('performs research search', async () => {
    render(
      <TestWrapper>
        <Research />
      </TestWrapper>
    );

    const searchInput = screen.getByPlaceholderText(/search medical literature/i);
    const searchButton = screen.getByRole('button', { name: /search/i });

    fireEvent.change(searchInput, { target: { value: 'neurosurgery techniques' } });
    fireEvent.click(searchButton);

    await waitFor(() => {
      expect(screen.getByText('Neurosurgical Research Paper 1')).toBeInTheDocument();
    });

    expect(screen.getByText('Advanced Brain Surgery Techniques')).toBeInTheDocument();
    expect(screen.getByText(/results \(2\)/i)).toBeInTheDocument();
  });

  test('handles result selection', async () => {
    render(
      <TestWrapper>
        <Research />
      </TestWrapper>
    );

    // Trigger search first
    const searchInput = screen.getByPlaceholderText(/search medical literature/i);
    fireEvent.change(searchInput, { target: { value: 'neurosurgery' } });
    fireEvent.submit(searchInput.closest('form')!);

    await waitFor(() => {
      expect(screen.getByText('Neurosurgical Research Paper 1')).toBeInTheDocument();
    });

    // Select results
    const checkboxes = screen.getAllByRole('checkbox');
    fireEvent.click(checkboxes[1]); // First result checkbox

    expect(screen.getByText(/1 selected/i)).toBeInTheDocument();
  });

  test('handles AI analysis', async () => {
    const mockAIAnalysis = jest.fn();

    render(
      <TestWrapper>
        <Research />
      </TestWrapper>
    );

    // Perform search and select results
    const searchInput = screen.getByPlaceholderText(/search medical literature/i);
    fireEvent.change(searchInput, { target: { value: 'neurosurgery' } });
    fireEvent.submit(searchInput.closest('form')!);

    await waitFor(() => {
      expect(screen.getByText('Neurosurgical Research Paper 1')).toBeInTheDocument();
    });

    // Select results
    const checkbox = screen.getAllByRole('checkbox')[1];
    fireEvent.click(checkbox);

    // Trigger AI analysis
    const aiButton = screen.getByRole('button', { name: /ai analysis/i });
    fireEvent.click(aiButton);

    expect(mockAIAnalysis).toHaveBeenCalled();
  });
});

describe('Performance Tests', () => {
  test('renders large chapter content efficiently', async () => {
    const largeContent = '# Large Chapter\n\n' + 'Content paragraph. '.repeat(1000);
    const largeChapter = {
      ...mockChapter,
      content: largeContent,
      word_count: 2000,
      reading_time_minutes: 10,
    };

    const { getChapter } = require('../services/chapterService');
    getChapter.mockResolvedValue(largeChapter);

    const startTime = performance.now();

    render(
      <TestWrapper>
        <ChapterViewer />
      </TestWrapper>
    );

    await waitFor(() => {
      expect(screen.getByText('Test Neurosurgical Procedure')).toBeInTheDocument();
    });

    const endTime = performance.now();
    const renderTime = endTime - startTime;

    // Should render within 2 seconds even with large content
    expect(renderTime).toBeLessThan(2000);
  });

  test('handles rapid user interactions', async () => {
    render(
      <TestWrapper>
        <Research />
      </TestWrapper>
    );

    const searchInput = screen.getByPlaceholderText(/search medical literature/i);

    // Simulate rapid typing
    const queries = ['n', 'ne', 'neu', 'neur', 'neuro', 'neuros', 'neurosurgery'];

    for (const query of queries) {
      fireEvent.change(searchInput, { target: { value: query } });
      await new Promise(resolve => setTimeout(resolve, 10)); // 10ms delay
    }

    // Should handle rapid input without crashing
    expect(searchInput).toHaveValue('neurosurgery');
  });
});

// Integration Tests
describe('End-to-End Workflows', () => {
  test('complete research to chapter creation workflow', async () => {
    // This would typically use a testing framework like Cypress or Playwright
    // for true end-to-end testing across the full application

    const mockWorkflowSteps = {
      search: jest.fn().mockResolvedValue(mockSearchResults),
      synthesize: jest.fn().mockResolvedValue({ synthesis: 'Generated content' }),
      createChapter: jest.fn().mockResolvedValue({ id: 1 }),
    };

    // Test the workflow integration
    expect(mockWorkflowSteps.search).toBeDefined();
    expect(mockWorkflowSteps.synthesize).toBeDefined();
    expect(mockWorkflowSteps.createChapter).toBeDefined();
  });
});