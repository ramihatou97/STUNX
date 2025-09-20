/**
 * Enhanced Research Analytics Component
 * Displays advanced PubMed analytics, citation networks, and research trends
 */

import React, { useState, useEffect } from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  Grid,
  Button,
  TextField,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  Chip,
  CircularProgress,
  Alert,
  Tabs,
  Tab,
  List,
  ListItem,
  ListItemText,
  LinearProgress
} from '@mui/material';
import {
  TrendingUp,
  AccountTree,
  Assessment,
  Notifications,
  Star,
  Share,
  Article
} from '@mui/icons-material';
import apiService from '../../services/api';

interface CitationNetwork {
  papers: any[];
  network_metrics: {
    total_nodes: number;
    total_edges: number;
    density: number;
    average_degree: number;
  };
  research_clusters: any[];
  key_authors: any[];
}

interface ResearchTrend {
  specialty: string;
  momentum: number;
  publication_trends: any;
  emerging_topics: string[];
  collaboration_networks: any;
}

interface ResearchAlert {
  alert_id: string;
  topic: string;
  alert_type: string;
  frequency: string;
  active: boolean;
  last_triggered: string | null;
}

interface QualityScore {
  quality_score: number;
  relevance_score: number;
  combined_score: number;
  recommendation_reasons: string[];
}

const EnhancedResearchAnalytics: React.FC = () => {
  const [activeTab, setActiveTab] = useState(0);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Citation Network State
  const [citationNetwork, setCitationNetwork] = useState<CitationNetwork | null>(null);
  const [citationTopic, setCitationTopic] = useState('');

  // Research Trends State
  const [researchTrends, setResearchTrends] = useState<ResearchTrend | null>(null);
  const [trendsSpecialty, setTrendsSpecialty] = useState('neurosurgery');

  // Research Alerts State
  const [alerts, setAlerts] = useState<ResearchAlert[]>([]);
  const [newAlertTopic, setNewAlertTopic] = useState('');
  const [newAlertType, setNewAlertType] = useState('new_publication');

  // Enhanced Recommendations State
  const [recommendations, setRecommendations] = useState<any[]>([]);
  const [userInterests, setUserInterests] = useState<string[]>(['neurosurgery', 'brain tumors']);

  useEffect(() => {
    fetchAlerts();
  }, []);

  const fetchCitationNetwork = async () => {
    if (!citationTopic.trim()) return;

    setLoading(true);
    setError(null);

    try {
      const response = await apiService.request('/api/v1/enhanced-research/analytics/citation-network', {
        method: 'POST',
        body: JSON.stringify({
          topic: citationTopic,
          max_papers: 50,
          years_back: 5
        })
      });

      setCitationNetwork(response.analysis);
    } catch (err: any) {
      setError(err.message || 'Failed to fetch citation network');
    } finally {
      setLoading(false);
    }
  };

  const fetchResearchTrends = async () => {
    setLoading(true);
    setError(null);

    try {
      const response = await apiService.request('/api/v1/enhanced-research/analytics/research-trends', {
        method: 'POST',
        body: JSON.stringify({
          specialty: trendsSpecialty,
          years: 10
        })
      });

      setResearchTrends(response.trends);
    } catch (err: any) {
      setError(err.message || 'Failed to fetch research trends');
    } finally {
      setLoading(false);
    }
  };

  const fetchAlerts = async () => {
    try {
      const response = await apiService.request('/api/v1/enhanced-research/alerts');
      setAlerts(response.alerts);
    } catch (err: any) {
      console.error('Failed to fetch alerts:', err);
    }
  };

  const createAlert = async () => {
    if (!newAlertTopic.trim()) return;

    try {
      await apiService.request('/api/v1/enhanced-research/alerts/create', {
        method: 'POST',
        body: JSON.stringify({
          topic: newAlertTopic,
          alert_type: newAlertType,
          frequency: 'weekly'
        })
      });

      setNewAlertTopic('');
      fetchAlerts();
    } catch (err: any) {
      setError(err.message || 'Failed to create alert');
    }
  };

  const fetchEnhancedRecommendations = async () => {
    setLoading(true);
    setError(null);

    try {
      const response = await apiService.request('/api/v1/enhanced-research/recommendations/enhanced', {
        method: 'POST',
        body: JSON.stringify({
          user_interests: userInterests,
          max_results: 20,
          quality_threshold: 0.6
        })
      });

      setRecommendations(response.recommendations);
    } catch (err: any) {
      setError(err.message || 'Failed to fetch recommendations');
    } finally {
      setLoading(false);
    }
  };

  const renderCitationNetworkTab = () => (
    <Box>
      <Grid container spacing={3}>
        <Grid item xs={12}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                <AccountTree sx={{ mr: 1, verticalAlign: 'middle' }} />
                Citation Network Analysis
              </Typography>

              <Box display="flex" gap={2} mb={3}>
                <TextField
                  label="Research Topic"
                  value={citationTopic}
                  onChange={(e) => setCitationTopic(e.target.value)}
                  fullWidth
                  placeholder="e.g., glioblastoma treatment"
                />
                <Button
                  variant="contained"
                  onClick={fetchCitationNetwork}
                  disabled={loading || !citationTopic.trim()}
                >
                  Analyze
                </Button>
              </Box>

              {loading && <CircularProgress />}

              {citationNetwork && (
                <Grid container spacing={2}>
                  <Grid item xs={12} md={6}>
                    <Typography variant="subtitle1" gutterBottom>Network Metrics</Typography>
                    <Box>
                      <Typography>Papers: {citationNetwork.network_metrics.total_nodes}</Typography>
                      <Typography>Citations: {citationNetwork.network_metrics.total_edges}</Typography>
                      <Typography>Network Density: {(citationNetwork.network_metrics.density * 100).toFixed(1)}%</Typography>
                    </Box>
                  </Grid>

                  <Grid item xs={12} md={6}>
                    <Typography variant="subtitle1" gutterBottom>Research Clusters</Typography>
                    {citationNetwork.research_clusters.slice(0, 3).map((cluster, idx) => (
                      <Chip
                        key={idx}
                        label={`Cluster ${cluster.cluster_id} (${cluster.paper_count} papers)`}
                        variant="outlined"
                        sx={{ mr: 1, mb: 1 }}
                      />
                    ))}
                  </Grid>
                </Grid>
              )}
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Box>
  );

  const renderResearchTrendsTab = () => (
    <Box>
      <Grid container spacing={3}>
        <Grid item xs={12}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                <TrendingUp sx={{ mr: 1, verticalAlign: 'middle' }} />
                Research Trends Analysis
              </Typography>

              <Box display="flex" gap={2} mb={3}>
                <FormControl sx={{ minWidth: 200 }}>
                  <InputLabel>Specialty</InputLabel>
                  <Select
                    value={trendsSpecialty}
                    onChange={(e) => setTrendsSpecialty(e.target.value)}
                  >
                    <MenuItem value="neurosurgery">Neurosurgery</MenuItem>
                    <MenuItem value="oncology">Neuro-oncology</MenuItem>
                    <MenuItem value="vascular">Vascular</MenuItem>
                    <MenuItem value="pediatric">Pediatric</MenuItem>
                  </Select>
                </FormControl>
                <Button
                  variant="contained"
                  onClick={fetchResearchTrends}
                  disabled={loading}
                >
                  Analyze Trends
                </Button>
              </Box>

              {loading && <CircularProgress />}

              {researchTrends && (
                <Grid container spacing={2}>
                  <Grid item xs={12} md={6}>
                    <Typography variant="subtitle1" gutterBottom>
                      Trend Momentum: {(researchTrends.momentum * 100).toFixed(1)}%
                    </Typography>
                    <LinearProgress
                      variant="determinate"
                      value={researchTrends.momentum * 100}
                      sx={{ height: 10, borderRadius: 5, mb: 2 }}
                    />

                    <Typography variant="subtitle1" gutterBottom>Emerging Topics</Typography>
                    {researchTrends.emerging_topics.map((topic, idx) => (
                      <Chip
                        key={idx}
                        label={topic}
                        color="primary"
                        variant="outlined"
                        sx={{ mr: 1, mb: 1 }}
                      />
                    ))}
                  </Grid>

                  <Grid item xs={12} md={6}>
                    <Typography variant="subtitle1" gutterBottom>Publication Trends</Typography>
                    <Typography variant="body2">
                      Velocity: {researchTrends.publication_trends?.velocity || 'N/A'} papers/month
                    </Typography>
                    <Typography variant="body2">
                      Acceleration: {((researchTrends.publication_trends?.acceleration || 0) * 100).toFixed(1)}%
                    </Typography>
                  </Grid>
                </Grid>
              )}
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Box>
  );

  const renderAlertsTab = () => (
    <Box>
      <Grid container spacing={3}>
        <Grid item xs={12}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                <Notifications sx={{ mr: 1, verticalAlign: 'middle' }} />
                Research Alerts
              </Typography>

              <Box display="flex" gap={2} mb={3}>
                <TextField
                  label="Alert Topic"
                  value={newAlertTopic}
                  onChange={(e) => setNewAlertTopic(e.target.value)}
                  placeholder="e.g., deep brain stimulation"
                  sx={{ flexGrow: 1 }}
                />
                <FormControl sx={{ minWidth: 150 }}>
                  <InputLabel>Alert Type</InputLabel>
                  <Select
                    value={newAlertType}
                    onChange={(e) => setNewAlertType(e.target.value)}
                  >
                    <MenuItem value="new_publication">New Publications</MenuItem>
                    <MenuItem value="trend_change">Trend Changes</MenuItem>
                    <MenuItem value="breakthrough">Breakthroughs</MenuItem>
                  </Select>
                </FormControl>
                <Button
                  variant="contained"
                  onClick={createAlert}
                  disabled={!newAlertTopic.trim()}
                >
                  Create Alert
                </Button>
              </Box>

              <List>
                {alerts.map((alert) => (
                  <ListItem key={alert.alert_id} divider>
                    <ListItemText
                      primary={alert.topic}
                      secondary={`${alert.alert_type} • ${alert.frequency} • ${alert.active ? 'Active' : 'Inactive'}`}
                    />
                    <Chip
                      label={alert.active ? 'Active' : 'Inactive'}
                      color={alert.active ? 'success' : 'default'}
                      size="small"
                    />
                  </ListItem>
                ))}
              </List>
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Box>
  );

  const renderRecommendationsTab = () => (
    <Box>
      <Grid container spacing={3}>
        <Grid item xs={12}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                <Star sx={{ mr: 1, verticalAlign: 'middle' }} />
                Enhanced Research Recommendations
              </Typography>

              <Box display="flex" gap={2} mb={3}>
                <TextField
                  label="Research Interests (comma-separated)"
                  value={userInterests.join(', ')}
                  onChange={(e) => setUserInterests(e.target.value.split(',').map(s => s.trim()))}
                  fullWidth
                  placeholder="neurosurgery, glioblastoma, stereotactic surgery"
                />
                <Button
                  variant="contained"
                  onClick={fetchEnhancedRecommendations}
                  disabled={loading}
                >
                  Get Recommendations
                </Button>
              </Box>

              {loading && <CircularProgress />}

              <Grid container spacing={2}>
                {recommendations.map((rec, idx) => (
                  <Grid item xs={12} key={idx}>
                    <Card variant="outlined">
                      <CardContent>
                        <Box display="flex" justifyContent="between" alignItems="start">
                          <Box sx={{ flexGrow: 1 }}>
                            <Typography variant="subtitle1" gutterBottom>
                              {rec.title}
                            </Typography>
                            <Typography variant="body2" color="textSecondary" gutterBottom>
                              {rec.authors} • {rec.journal} • {rec.pub_date}
                            </Typography>
                            <Typography variant="body2" sx={{ mb: 2 }}>
                              {rec.abstract?.substring(0, 200)}...
                            </Typography>

                            <Box display="flex" gap={1} mb={1}>
                              <Chip
                                label={`Quality: ${(rec.quality_score * 100).toFixed(0)}%`}
                                color={rec.quality_score > 0.8 ? 'success' : rec.quality_score > 0.6 ? 'warning' : 'default'}
                                size="small"
                              />
                              <Chip
                                label={`Relevance: ${(rec.relevance_score * 100).toFixed(0)}%`}
                                color="primary"
                                size="small"
                              />
                            </Box>

                            {rec.recommendation_reasons && (
                              <Box>
                                <Typography variant="caption" display="block" gutterBottom>
                                  Why recommended:
                                </Typography>
                                {rec.recommendation_reasons.slice(0, 2).map((reason: string, ridx: number) => (
                                  <Typography key={ridx} variant="caption" display="block" color="textSecondary">
                                    • {reason}
                                  </Typography>
                                ))}
                              </Box>
                            )}
                          </Box>

                          <Box display="flex" flexDirection="column" gap={1}>
                            <Button size="small" startIcon={<Article />}>
                              View
                            </Button>
                            <Button size="small" startIcon={<Share />}>
                              Share
                            </Button>
                          </Box>
                        </Box>
                      </CardContent>
                    </Card>
                  </Grid>
                ))}
              </Grid>
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Box>
  );

  return (
    <Box sx={{ p: 3 }}>
      <Typography variant="h4" gutterBottom>
        <Assessment sx={{ mr: 2, verticalAlign: 'middle' }} />
        Enhanced Research Analytics
      </Typography>

      {error && (
        <Alert severity="error" sx={{ mb: 3 }} onClose={() => setError(null)}>
          {error}
        </Alert>
      )}

      <Box sx={{ borderBottom: 1, borderColor: 'divider', mb: 3 }}>
        <Tabs value={activeTab} onChange={(_, newValue) => setActiveTab(newValue)}>
          <Tab label="Citation Networks" icon={<AccountTree />} />
          <Tab label="Research Trends" icon={<TrendingUp />} />
          <Tab label="Research Alerts" icon={<Notifications />} />
          <Tab label="Recommendations" icon={<Star />} />
        </Tabs>
      </Box>

      {activeTab === 0 && renderCitationNetworkTab()}
      {activeTab === 1 && renderResearchTrendsTab()}
      {activeTab === 2 && renderAlertsTab()}
      {activeTab === 3 && renderRecommendationsTab()}
    </Box>
  );
};

export default EnhancedResearchAnalytics;