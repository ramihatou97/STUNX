/**
 * Enhanced Chapter Viewer with AI-powered features
 * Real-time updates, conflict resolution, and interactive research
 */

import React, { useState, useEffect, useMemo } from 'react';
import {
  Box, Paper, Typography, Chip, Button, IconButton, Grid,
  Accordion, AccordionSummary, AccordionDetails, Alert,
  Dialog, DialogTitle, DialogContent, DialogActions,
  LinearProgress, Tooltip, Menu, MenuItem, Fab
} from '@mui/material';
import {
  ExpandMore as ExpandMoreIcon,
  Edit as EditIcon,
  Share as ShareIcon,
  Bookmark as BookmarkIcon,
  Search as SearchIcon,
  Timeline as TimelineIcon,
  AutoAwesome as AIIcon,
  Refresh as RefreshIcon,
  MoreVert as MoreIcon
} from '@mui/icons-material';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { useParams, useNavigate } from 'react-router-dom';
import { format } from 'date-fns';

// Custom hooks
import { useAuth } from '../../contexts/AuthContext';
import { useWebSocket } from '../../contexts/WebSocketContext';

// Components
import ConflictResolutionDialog from './ConflictResolutionDialog';
import AIInsightsPanel from './AIInsightsPanel';
import ReferencesPanel from './ReferencesPanel';
import EditHistoryPanel from './EditHistoryPanel';
import LoadingScreen from '../common/LoadingScreen';

// Services
import * as chapterService from '../../services/chapterService';
import * as researchService from '../../services/researchService';

// Types
interface Chapter {
  id: string;
  title: string;
  content: string;
  summary: string;
  tags: string[];
  status: 'draft' | 'published' | 'archived';
  version: number;
  created_at: string;
  updated_at: string;
  author: {
    id: string;
    name: string;
    title: string;
  };
  sections: ChapterSection[];
  references: Reference[];
  ai_insights?: AIInsight[];
  conflicts?: ConflictData[];
  thought_stream?: ThoughtStreamEntry[];
}

interface ChapterSection {
  id: string;
  title: string;
  content: string;
  order: number;
  confidence_score: number;
  last_verified: string;
  sources: string[];
}

interface ConflictData {
  id: string;
  section_id: string;
  type: 'contradiction' | 'outdated' | 'uncertain';
  description: string;
  suggestions: string[];
  severity: 'low' | 'medium' | 'high';
}

interface AIInsight {
  id: string;
  type: 'suggestion' | 'improvement' | 'research_gap';
  content: string;
  relevance_score: number;
  generated_at: string;
}

const ChapterViewer: React.FC = () => {
  const { chapterId } = useParams<{ chapterId: string }>();
  const navigate = useNavigate();
  const { user } = useAuth();
  const { subscribeToChapter, unsubscribeFromChapter } = useWebSocket();
  const queryClient = useQueryClient();

  // State
  const [selectedSection, setSelectedSection] = useState<string | null>(null);
  const [showConflicts, setShowConflicts] = useState(false);
  const [showAIInsights, setShowAIInsights] = useState(false);
  const [showReferences, setShowReferences] = useState(false);
  const [showHistory, setShowHistory] = useState(false);
  const [anchorEl, setAnchorEl] = useState<null | HTMLElement>(null);
  const [isBookmarked, setIsBookmarked] = useState(false);
  const [autoRefresh, setAutoRefresh] = useState(true);

  // Queries
  const {
    data: chapter,
    isLoading,
    error,
    refetch
  } = useQuery({
    queryKey: ['chapter', chapterId],
    queryFn: () => chapterService.getChapter(chapterId!),
    enabled: !!chapterId,
    refetchInterval: autoRefresh ? 30000 : false, // Auto-refresh every 30 seconds
  });

  const {
    data: relatedResearch,
    isLoading: isLoadingResearch
  } = useQuery({
    queryKey: ['related-research', chapterId],
    queryFn: () => researchService.getRelatedResearch(chapterId!),
    enabled: !!chapterId && showAIInsights,
  });

  // Mutations
  const refreshChapterMutation = useMutation({
    mutationFn: () => chapterService.refreshChapterContent(chapterId!),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['chapter', chapterId] });
    },
  });

  const bookmarkMutation = useMutation({
    mutationFn: (bookmarked: boolean) =>
      chapterService.setBookmark(chapterId!, bookmarked),
    onSuccess: (_, bookmarked) => {
      setIsBookmarked(bookmarked);
    },
  });

  // WebSocket subscription for real-time updates
  useEffect(() => {
    if (chapterId && user) {
      subscribeToChapter(chapterId, (update) => {
        queryClient.setQueryData(['chapter', chapterId], (oldData: Chapter | undefined) => {
          if (!oldData) return oldData;
          return { ...oldData, ...update };
        });
      });

      return () => unsubscribeFromChapter(chapterId);
    }
  }, [chapterId, user, subscribeToChapter, unsubscribeFromChapter, queryClient]);

  // Computed values
  const hasConflicts = useMemo(() =>
    chapter?.conflicts && chapter.conflicts.length > 0, [chapter]);

  const overallConfidence = useMemo(() => {
    if (!chapter?.sections) return 0;
    const scores = chapter.sections.map(s => s.confidence_score);
    return scores.reduce((sum, score) => sum + score, 0) / scores.length;
  }, [chapter]);

  const lastUpdated = useMemo(() => {
    if (!chapter) return '';
    return format(new Date(chapter.updated_at), 'PPp');
  }, [chapter]);

  // Event handlers
  const handleEdit = () => {
    navigate(`/chapter/${chapterId}/edit`);
  };

  const handleShare = () => {
    navigator.clipboard.writeText(window.location.href);
    // Show success message
  };

  const handleRefresh = () => {
    refreshChapterMutation.mutate();
  };

  const handleBookmark = () => {
    bookmarkMutation.mutate(!isBookmarked);
  };

  const handleMenuClick = (event: React.MouseEvent<HTMLElement>) => {
    setAnchorEl(event.currentTarget);
  };

  const handleMenuClose = () => {
    setAnchorEl(null);
  };

  if (isLoading) return <LoadingScreen />;
  if (error) return <Alert severity="error">Failed to load chapter</Alert>;
  if (!chapter) return <Alert severity="warning">Chapter not found</Alert>;

  return (
    <Box>
      {/* Chapter Header */}
      <Paper sx={{ p: 3, mb: 3 }}>
        <Grid container alignItems="center" spacing={2}>
          <Grid item xs>
            <Typography variant="h4" gutterBottom>
              {chapter.title}
            </Typography>
            <Typography variant="body2" color="text.secondary" gutterBottom>
              By {chapter.author.name} • Last updated {lastUpdated} • Version {chapter.version}
            </Typography>
            <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap', mt: 1 }}>
              {chapter.tags.map((tag) => (
                <Chip key={tag} label={tag} size="small" />
              ))}
              <Chip
                label={`${(overallConfidence * 100).toFixed(0)}% confidence`}
                color={overallConfidence > 0.8 ? 'success' : overallConfidence > 0.6 ? 'warning' : 'error'}
                size="small"
              />
              {hasConflicts && (
                <Chip
                  label={`${chapter.conflicts!.length} conflicts`}
                  color="error"
                  size="small"
                  onClick={() => setShowConflicts(true)}
                />
              )}
            </Box>
          </Grid>
          <Grid item>
            <Box sx={{ display: 'flex', gap: 1 }}>
              <Tooltip title="Edit Chapter">
                <IconButton onClick={handleEdit} color="primary">
                  <EditIcon />
                </IconButton>
              </Tooltip>
              <Tooltip title="Share Chapter">
                <IconButton onClick={handleShare}>
                  <ShareIcon />
                </IconButton>
              </Tooltip>
              <Tooltip title={isBookmarked ? "Remove Bookmark" : "Add Bookmark"}>
                <IconButton
                  onClick={handleBookmark}
                  color={isBookmarked ? "warning" : "default"}
                >
                  <BookmarkIcon />
                </IconButton>
              </Tooltip>
              <Tooltip title="Refresh Content">
                <IconButton
                  onClick={handleRefresh}
                  disabled={refreshChapterMutation.isPending}
                >
                  <RefreshIcon />
                </IconButton>
              </Tooltip>
              <IconButton onClick={handleMenuClick}>
                <MoreIcon />
              </IconButton>
            </Box>
          </Grid>
        </Grid>

        {/* Progress bar for refresh */}
        {refreshChapterMutation.isPending && (
          <LinearProgress sx={{ mt: 2 }} />
        )}
      </Paper>

      {/* Chapter Summary */}
      {chapter.summary && (
        <Paper sx={{ p: 3, mb: 3 }}>
          <Typography variant="h6" gutterBottom>
            Summary
          </Typography>
          <Typography variant="body1">
            {chapter.summary}
          </Typography>
        </Paper>
      )}

      {/* Chapter Sections */}
      <Paper sx={{ p: 3, mb: 3 }}>
        <Typography variant="h6" gutterBottom>
          Chapter Content
        </Typography>
        {chapter.sections.map((section, index) => (
          <Accordion
            key={section.id}
            expanded={selectedSection === section.id}
            onChange={() => setSelectedSection(
              selectedSection === section.id ? null : section.id
            )}
          >
            <AccordionSummary
              expandIcon={<ExpandMoreIcon />}
              sx={{
                backgroundColor: section.confidence_score < 0.7 ? 'warning.light' : 'inherit'
              }}
            >
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, width: '100%' }}>
                <Typography variant="h6">
                  {section.title}
                </Typography>
                <Chip
                  label={`${(section.confidence_score * 100).toFixed(0)}%`}
                  color={section.confidence_score > 0.8 ? 'success' :
                         section.confidence_score > 0.6 ? 'warning' : 'error'}
                  size="small"
                />
                <Typography variant="caption" color="text.secondary" sx={{ ml: 'auto' }}>
                  Verified {format(new Date(section.last_verified), 'MMM dd, yyyy')}
                </Typography>
              </Box>
            </AccordionSummary>
            <AccordionDetails>
              <Typography
                variant="body1"
                dangerouslySetInnerHTML={{ __html: section.content }}
                sx={{ lineHeight: 1.8 }}
              />
              {section.sources.length > 0 && (
                <Box sx={{ mt: 2 }}>
                  <Typography variant="subtitle2" gutterBottom>
                    Sources:
                  </Typography>
                  <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap' }}>
                    {section.sources.map((source, idx) => (
                      <Chip key={idx} label={source} size="small" variant="outlined" />
                    ))}
                  </Box>
                </Box>
              )}
            </AccordionDetails>
          </Accordion>
        ))}
      </Paper>

      {/* Floating Action Buttons */}
      <Box sx={{ position: 'fixed', bottom: 24, right: 24, display: 'flex', flexDirection: 'column', gap: 1 }}>
        <Fab
          color="primary"
          onClick={() => setShowAIInsights(true)}
          size="medium"
        >
          <AIIcon />
        </Fab>
        <Fab
          onClick={() => setShowReferences(true)}
          size="medium"
        >
          <SearchIcon />
        </Fab>
        <Fab
          onClick={() => setShowHistory(true)}
          size="medium"
        >
          <TimelineIcon />
        </Fab>
      </Box>

      {/* Context Menu */}
      <Menu
        anchorEl={anchorEl}
        open={Boolean(anchorEl)}
        onClose={handleMenuClose}
      >
        <MenuItem onClick={() => { setShowConflicts(true); handleMenuClose(); }}>
          View Conflicts
        </MenuItem>
        <MenuItem onClick={() => { setShowAIInsights(true); handleMenuClose(); }}>
          AI Insights
        </MenuItem>
        <MenuItem onClick={() => { setShowReferences(true); handleMenuClose(); }}>
          References
        </MenuItem>
        <MenuItem onClick={() => { setShowHistory(true); handleMenuClose(); }}>
          Edit History
        </MenuItem>
        <MenuItem onClick={() => { setAutoRefresh(!autoRefresh); handleMenuClose(); }}>
          {autoRefresh ? 'Disable' : 'Enable'} Auto-refresh
        </MenuItem>
      </Menu>

      {/* Dialogs */}
      {showConflicts && chapter.conflicts && (
        <ConflictResolutionDialog
          open={showConflicts}
          onClose={() => setShowConflicts(false)}
          conflicts={chapter.conflicts}
          chapterId={chapterId!}
        />
      )}

      {showAIInsights && (
        <AIInsightsPanel
          open={showAIInsights}
          onClose={() => setShowAIInsights(false)}
          chapter={chapter}
          relatedResearch={relatedResearch}
          isLoading={isLoadingResearch}
        />
      )}

      {showReferences && (
        <ReferencesPanel
          open={showReferences}
          onClose={() => setShowReferences(false)}
          references={chapter.references}
          chapterId={chapterId!}
        />
      )}

      {showHistory && (
        <EditHistoryPanel
          open={showHistory}
          onClose={() => setShowHistory(false)}
          chapterId={chapterId!}
        />
      )}
    </Box>
  );
};

export default ChapterViewer;