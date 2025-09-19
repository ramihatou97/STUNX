/**
 * Advanced Academic Research Dashboard
 * Multi-API integration with real-time research capabilities
 */

import React, { useState, useEffect } from 'react';
import {
  Grid, Paper, Typography, Box, TextField, Button, Chip,
  Card, CardContent, CardActions, Tab, Tabs, CircularProgress,
  Alert, Accordion, AccordionSummary, AccordionDetails,
  FormControl, InputLabel, Select, MenuItem, Switch,
  FormControlLabel, Divider, List, ListItem, ListItemText,
  ListItemIcon, Badge
} from '@mui/material';
import {
  Search as SearchIcon,
  FilterList as FilterIcon,
  Download as DownloadIcon,
  OpenInNew as OpenIcon,
  Bookmark as BookmarkIcon,
  Science as ScienceIcon,
  School as SchoolIcon,
  LocalHospital as MedicalIcon,
  Timeline as TimelineIcon,
  ExpandMore as ExpandMoreIcon,
  AutoAwesome as AIIcon
} from '@mui/icons-material';
import { useQuery, useMutation } from '@tanstack/react-query';

// Services
import * as researchService from '../../services/researchService';

// Types
interface ResearchResult {
  id: string;
  title: string;
  authors: string[];
  abstract: string;
  journal: string;
  year: number;
  doi?: string;
  pmid?: string;
  url?: string;
  relevance_score: number;
  source: 'pubmed' | 'semantic_scholar' | 'perplexity' | 'elsevier';
  created_at: string;
}

interface ResearchFilters {
  sources: string[];
  dateRange: { start: number; end: number };
  minRelevanceScore: number;
  journalTypes: string[];
  hasFullText: boolean;
}

const AcademicDashboard: React.FC = () => {
  // State
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedTab, setSelectedTab] = useState(0);
  const [filters, setFilters] = useState<ResearchFilters>({
    sources: ['pubmed', 'semantic_scholar'],
    dateRange: { start: 2020, end: new Date().getFullYear() },
    minRelevanceScore: 0.7,
    journalTypes: ['all'],
    hasFullText: false
  });
  const [savedSearches, setSavedSearches] = useState<string[]>([]);
  const [aiInsightsEnabled, setAiInsightsEnabled] = useState(true);

  // Queries
  const {
    data: searchResults,
    isLoading: isSearching,
    error: searchError,
    refetch: executeSearch
  } = useQuery({
    queryKey: ['research-search', searchQuery, filters],
    queryFn: () => researchService.searchLiterature(searchQuery, filters),
    enabled: false, // Manual trigger
  });

  const {
    data: aiInsights,
    isLoading: isLoadingInsights
  } = useQuery({
    queryKey: ['ai-insights', searchQuery],
    queryFn: () => researchService.getAIInsights(searchQuery),
    enabled: aiInsightsEnabled && !!searchQuery && searchQuery.length > 10,
  });

  const {
    data: savedResults,
    refetch: refreshSaved
  } = useQuery({
    queryKey: ['saved-research'],
    queryFn: () => researchService.getSavedResearch(),
  });

  // Mutations
  const saveSearchMutation = useMutation({
    mutationFn: (query: string) => researchService.saveSearch(query),
    onSuccess: () => {
      setSavedSearches(prev => [...prev, searchQuery]);
    },
  });

  const bookmarkMutation = useMutation({
    mutationFn: (resultId: string) => researchService.bookmarkResult(resultId),
    onSuccess: () => {
      refreshSaved();
    },
  });

  // Event handlers
  const handleSearch = () => {
    if (searchQuery.trim()) {
      executeSearch();
    }
  };

  const handleKeyPress = (event: React.KeyboardEvent) => {
    if (event.key === 'Enter') {
      handleSearch();
    }
  };

  const handleFilterChange = (newFilters: Partial<ResearchFilters>) => {
    setFilters(prev => ({ ...prev, ...newFilters }));
  };

  const handleSaveSearch = () => {
    if (searchQuery.trim() && !savedSearches.includes(searchQuery)) {
      saveSearchMutation.mutate(searchQuery);
    }
  };

  const handleBookmark = (resultId: string) => {
    bookmarkMutation.mutate(resultId);
  };

  const handleTabChange = (_: React.SyntheticEvent, newValue: number) => {
    setSelectedTab(newValue);
  };

  // Render result card
  const renderResultCard = (result: ResearchResult) => (
    <Card key={result.id} sx={{ mb: 2, border: '1px solid #e0e0e0' }}>
      <CardContent>
        <Box sx={{ display: 'flex', alignItems: 'flex-start', gap: 2 }}>
          <Box sx={{ flexGrow: 1 }}>
            <Typography variant="h6" gutterBottom>
              {result.title}
            </Typography>
            <Typography variant="body2" color="text.secondary" gutterBottom>
              {result.authors.join(', ')} • {result.journal} ({result.year})
            </Typography>
            <Typography variant="body2" sx={{ mt: 1, lineHeight: 1.6 }}>
              {result.abstract.length > 300
                ? `${result.abstract.substring(0, 300)}...`
                : result.abstract}
            </Typography>
            <Box sx={{ mt: 2, display: 'flex', gap: 1, flexWrap: 'wrap' }}>
              <Chip
                label={result.source.replace('_', ' ').toUpperCase()}
                size="small"
                color="primary"
                variant="outlined"
              />
              <Chip
                label={`${(result.relevance_score * 100).toFixed(0)}% relevant`}
                size="small"
                color={result.relevance_score > 0.8 ? 'success' : 'warning'}
              />
              {result.doi && (
                <Chip
                  label={`DOI: ${result.doi}`}
                  size="small"
                  variant="outlined"
                />
              )}
            </Box>
          </Box>
          <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
            <Button
              startIcon={<BookmarkIcon />}
              size="small"
              onClick={() => handleBookmark(result.id)}
            >
              Save
            </Button>
            {result.url && (
              <Button
                startIcon={<OpenIcon />}
                size="small"
                href={result.url}
                target="_blank"
                rel="noopener noreferrer"
              >
                Open
              </Button>
            )}
          </Box>
        </Box>
      </CardContent>
    </Card>
  );

  return (
    <Box>
      {/* Search Header */}
      <Paper sx={{ p: 3, mb: 3 }}>
        <Typography variant="h4" gutterBottom>
          Academic Research Dashboard
        </Typography>
        <Grid container spacing={2} alignItems="center">
          <Grid item xs={12} md={8}>
            <TextField
              fullWidth
              placeholder="Search medical literature, guidelines, and research..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              onKeyPress={handleKeyPress}
              variant="outlined"
              InputProps={{
                startAdornment: <SearchIcon sx={{ mr: 1, color: 'text.secondary' }} />,
              }}
            />
          </Grid>
          <Grid item xs={12} md={4}>
            <Box sx={{ display: 'flex', gap: 1 }}>
              <Button
                variant="contained"
                onClick={handleSearch}
                disabled={isSearching || !searchQuery.trim()}
                startIcon={isSearching ? <CircularProgress size={20} /> : <SearchIcon />}
              >
                Search
              </Button>
              <Button
                variant="outlined"
                onClick={handleSaveSearch}
                disabled={!searchQuery.trim() || savedSearches.includes(searchQuery)}
              >
                Save
              </Button>
            </Box>
          </Grid>
        </Grid>

        {/* AI Insights Toggle */}
        <Box sx={{ mt: 2 }}>
          <FormControlLabel
            control={
              <Switch
                checked={aiInsightsEnabled}
                onChange={(e) => setAiInsightsEnabled(e.target.checked)}
              />
            }
            label="Enable AI-powered research insights"
          />
        </Box>
      </Paper>

      {/* Filters */}
      <Paper sx={{ p: 2, mb: 3 }}>
        <Typography variant="h6" gutterBottom>
          Search Filters
        </Typography>
        <Grid container spacing={2}>
          <Grid item xs={12} md={3}>
            <FormControl fullWidth size="small">
              <InputLabel>Sources</InputLabel>
              <Select
                multiple
                value={filters.sources}
                onChange={(e) => handleFilterChange({ sources: e.target.value as string[] })}
                renderValue={(selected) => selected.join(', ')}
              >
                <MenuItem value="pubmed">PubMed</MenuItem>
                <MenuItem value="semantic_scholar">Semantic Scholar</MenuItem>
                <MenuItem value="perplexity">Perplexity</MenuItem>
                <MenuItem value="elsevier">Elsevier</MenuItem>
              </Select>
            </FormControl>
          </Grid>
          <Grid item xs={12} md={3}>
            <TextField
              fullWidth
              size="small"
              label="Min Relevance Score"
              type="number"
              value={filters.minRelevanceScore}
              onChange={(e) => handleFilterChange({ minRelevanceScore: parseFloat(e.target.value) })}
              inputProps={{ min: 0, max: 1, step: 0.1 }}
            />
          </Grid>
          <Grid item xs={12} md={3}>
            <TextField
              fullWidth
              size="small"
              label="From Year"
              type="number"
              value={filters.dateRange.start}
              onChange={(e) => handleFilterChange({
                dateRange: { ...filters.dateRange, start: parseInt(e.target.value) }
              })}
            />
          </Grid>
          <Grid item xs={12} md={3}>
            <TextField
              fullWidth
              size="small"
              label="To Year"
              type="number"
              value={filters.dateRange.end}
              onChange={(e) => handleFilterChange({
                dateRange: { ...filters.dateRange, end: parseInt(e.target.value) }
              })}
            />
          </Grid>
        </Grid>
      </Paper>

      {/* Results Tabs */}
      <Paper sx={{ mb: 3 }}>
        <Tabs value={selectedTab} onChange={handleTabChange} variant="fullWidth">
          <Tab
            icon={<SearchIcon />}
            label={`Search Results ${searchResults ? `(${searchResults.length})` : ''}`}
          />
          <Tab
            icon={<BookmarkIcon />}
            label={`Saved Research ${savedResults ? `(${savedResults.length})` : ''}`}
          />
          <Tab
            icon={<AIIcon />}
            label="AI Insights"
            disabled={!aiInsightsEnabled}
          />
        </Tabs>
      </Paper>

      {/* Tab Content */}
      {selectedTab === 0 && (
        <Box>
          {searchError && (
            <Alert severity="error" sx={{ mb: 2 }}>
              Failed to search: {searchError.message}
            </Alert>
          )}
          {isSearching && (
            <Box sx={{ display: 'flex', justifyContent: 'center', py: 4 }}>
              <CircularProgress />
            </Box>
          )}
          {searchResults && searchResults.length === 0 && (
            <Alert severity="info">No results found. Try adjusting your search terms or filters.</Alert>
          )}
          {searchResults && searchResults.map(renderResultCard)}
        </Box>
      )}

      {selectedTab === 1 && (
        <Box>
          {savedResults && savedResults.length === 0 && (
            <Alert severity="info">No saved research yet. Bookmark interesting papers from your searches.</Alert>
          )}
          {savedResults && savedResults.map(renderResultCard)}
        </Box>
      )}

      {selectedTab === 2 && aiInsightsEnabled && (
        <Box>
          {isLoadingInsights && (
            <Box sx={{ display: 'flex', justifyContent: 'center', py: 4 }}>
              <CircularProgress />
            </Box>
          )}
          {aiInsights && (
            <Paper sx={{ p: 3 }}>
              <Typography variant="h6" gutterBottom>
                AI-Powered Research Insights
              </Typography>
              {aiInsights.map((insight: any, index: number) => (
                <Accordion key={index}>
                  <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                    <Typography variant="subtitle1">
                      {insight.title}
                    </Typography>
                  </AccordionSummary>
                  <AccordionDetails>
                    <Typography variant="body2">
                      {insight.content}
                    </Typography>
                  </AccordionDetails>
                </Accordion>
              ))}
            </Paper>
          )}
        </Box>
      )}
    </Box>
  );
};

export default AcademicDashboard;