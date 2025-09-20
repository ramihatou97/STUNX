/**
 * Knowledge Graph Visualization Component
 * Interactive visualization of neurosurgical concept relationships
 */

import React, { useState, useEffect, useRef } from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  TextField,
  Button,
  Chip,
  IconButton,
  Menu,
  MenuItem,
  Slider,
  FormControl,
  InputLabel,
  Select,
  Tooltip,
  Paper,
  List,
  ListItem,
  ListItemText,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Tabs,
  Tab,
  Grid,
  Switch,
  FormControlLabel
} from '@mui/material';
import {
  AccountTree,
  ZoomIn,
  ZoomOut,
  Refresh,
  Settings,
  Download,
  Fullscreen,
  Info,
  FilterList,
  Search,
  CenterFocusStrong
} from '@mui/icons-material';
import * as d3 from 'd3';
import apiService from '../../services/api';

interface ConceptNode {
  id: string;
  name: string;
  type: string;
  group: number;
  size: number;
  color: string;
  x?: number;
  y?: number;
  fx?: number;
  fy?: number;
}

interface ConceptLink {
  source: string | ConceptNode;
  target: string | ConceptNode;
  type: string;
  strength: number;
  confidence: number;
  description: string;
}

interface ConceptGraph {
  nodes: ConceptNode[];
  links: ConceptLink[];
  centralConcept: string;
  metadata: {
    totalConcepts: number;
    relationshipTypes: string[];
    confidenceRange: [number, number];
  };
}

const KnowledgeGraphVisualization: React.FC = () => {
  const svgRef = useRef<SVGSVGElement>(null);
  const [currentGraph, setCurrentGraph] = useState<ConceptGraph | null>(null);
  const [centralConcept, setCentralConcept] = useState('glioblastoma');
  const [loading, setLoading] = useState(false);
  const [selectedNode, setSelectedNode] = useState<ConceptNode | null>(null);
  const [selectedLink, setSelectedLink] = useState<ConceptLink | null>(null);

  // Visualization controls
  const [zoomLevel, setZoomLevel] = useState(1);
  const [showLabels, setShowLabels] = useState(true);
  const [nodeSize, setNodeSize] = useState(10);
  const [linkStrength, setLinkStrength] = useState(0.5);
  const [confidenceFilter, setConfidenceFilter] = useState(0.0);
  const [selectedTypes, setSelectedTypes] = useState<string[]>([]);

  // UI state
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [nodeDetailsOpen, setNodeDetailsOpen] = useState(false);
  const [currentTab, setCurrentTab] = useState(0);

  // Graph simulation
  const simulationRef = useRef<d3.Simulation<ConceptNode, ConceptLink> | null>(null);

  useEffect(() => {
    if (centralConcept) {
      buildConceptGraph(centralConcept);
    }
  }, [centralConcept]);

  useEffect(() => {
    if (currentGraph) {
      renderGraph();
    }
  }, [currentGraph, showLabels, nodeSize, confidenceFilter, selectedTypes]);

  const buildConceptGraph = async (concept: string) => {
    setLoading(true);
    try {
      const response = await apiService.request('/api/v1/ai-knowledge/concept-graph', {
        method: 'POST',
        body: JSON.stringify({
          central_concept: concept,
          depth: 2
        })
      });

      if (response.status === 'success') {
        const graphData = response.concept_graph;

        // Transform data for D3
        const nodes: ConceptNode[] = [
          {
            id: graphData.central_concept,
            name: graphData.central_concept,
            type: 'central',
            group: 0,
            size: 20,
            color: '#ff4444'
          },
          ...graphData.related_concepts.map((concept: string, index: number) => ({
            id: concept,
            name: concept,
            type: getConceptType(concept),
            group: getConceptGroup(concept),
            size: 12,
            color: getConceptColor(concept)
          }))
        ];

        const links: ConceptLink[] = graphData.cross_references.map((ref: any) => ({
          source: ref.source_concept,
          target: ref.target_concept,
          type: ref.reference_type,
          strength: getStrengthValue(ref.strength),
          confidence: ref.confidence,
          description: ref.description
        }));

        const graph: ConceptGraph = {
          nodes,
          links,
          centralConcept: concept,
          metadata: {
            totalConcepts: nodes.length,
            relationshipTypes: [...new Set(links.map(l => l.type))],
            confidenceRange: [
              Math.min(...links.map(l => l.confidence)),
              Math.max(...links.map(l => l.confidence))
            ]
          }
        };

        setCurrentGraph(graph);
        setSelectedTypes(graph.metadata.relationshipTypes);
      }
    } catch (error) {
      console.error('Failed to build concept graph:', error);
    } finally {
      setLoading(false);
    }
  };

  const renderGraph = () => {
    if (!currentGraph || !svgRef.current) return;

    const svg = d3.select(svgRef.current);
    const width = 800;
    const height = 600;

    // Clear previous content
    svg.selectAll("*").remove();

    // Filter data based on controls
    const filteredLinks = currentGraph.links.filter(link =>
      link.confidence >= confidenceFilter &&
      selectedTypes.includes(link.type)
    );

    const connectedNodeIds = new Set<string>();
    filteredLinks.forEach(link => {
      connectedNodeIds.add(typeof link.source === 'string' ? link.source : link.source.id);
      connectedNodeIds.add(typeof link.target === 'string' ? link.target : link.target.id);
    });

    const filteredNodes = currentGraph.nodes.filter(node =>
      connectedNodeIds.has(node.id) || node.type === 'central'
    );

    // Create zoom behavior
    const zoom = d3.zoom<SVGSVGElement, unknown>()
      .scaleExtent([0.1, 10])
      .on('zoom', (event) => {
        const { transform } = event;
        svg.select('.graph-content').attr('transform', transform);
        setZoomLevel(transform.k);
      });

    svg.call(zoom);

    // Create main group
    const g = svg.append('g').attr('class', 'graph-content');

    // Create simulation
    const simulation = d3.forceSimulation<ConceptNode>(filteredNodes)
      .force('link', d3.forceLink<ConceptNode, ConceptLink>(filteredLinks)
        .id(d => d.id)
        .strength(linkStrength)
        .distance(100))
      .force('charge', d3.forceManyBody().strength(-300))
      .force('center', d3.forceCenter(width / 2, height / 2))
      .force('collision', d3.forceCollide().radius(d => d.size + 5));

    simulationRef.current = simulation;

    // Create arrow markers for directed links
    const defs = svg.append('defs');

    const arrowMarker = defs.append('marker')
      .attr('id', 'arrowhead')
      .attr('viewBox', '-0 -5 10 10')
      .attr('refX', 13)
      .attr('refY', 0)
      .attr('orient', 'auto')
      .attr('markerWidth', 13)
      .attr('markerHeight', 13)
      .attr('xoverflow', 'visible');

    arrowMarker.append('svg:path')
      .attr('d', 'M 0,-5 L 10 ,0 L 0,5')
      .attr('fill', '#999')
      .style('stroke', 'none');

    // Create links
    const links = g.append('g')
      .attr('class', 'links')
      .selectAll('line')
      .data(filteredLinks)
      .enter().append('line')
      .attr('stroke', d => getLinkColor(d.type))
      .attr('stroke-opacity', d => Math.max(0.3, d.confidence))
      .attr('stroke-width', d => Math.max(1, d.strength * 4))
      .attr('marker-end', 'url(#arrowhead)')
      .style('cursor', 'pointer')
      .on('click', (event, d) => {
        setSelectedLink(d);
        setNodeDetailsOpen(true);
      });

    // Create nodes
    const nodes = g.append('g')
      .attr('class', 'nodes')
      .selectAll('circle')
      .data(filteredNodes)
      .enter().append('circle')
      .attr('r', d => d.size * (nodeSize / 10))
      .attr('fill', d => d.color)
      .attr('stroke', '#fff')
      .attr('stroke-width', 2)
      .style('cursor', 'pointer')
      .on('click', (event, d) => {
        setSelectedNode(d);
        setNodeDetailsOpen(true);
      })
      .call(d3.drag<SVGCircleElement, ConceptNode>()
        .on('start', dragstarted)
        .on('drag', dragged)
        .on('end', dragended));

    // Add labels if enabled
    if (showLabels) {
      const labels = g.append('g')
        .attr('class', 'labels')
        .selectAll('text')
        .data(filteredNodes)
        .enter().append('text')
        .text(d => d.name)
        .style('font-size', '12px')
        .style('font-family', 'Arial, sans-serif')
        .style('fill', '#333')
        .style('text-anchor', 'middle')
        .style('pointer-events', 'none')
        .attr('dy', d => d.size * (nodeSize / 10) + 15);
    }

    // Simulation tick function
    simulation.on('tick', () => {
      links
        .attr('x1', d => (d.source as ConceptNode).x!)
        .attr('y1', d => (d.source as ConceptNode).y!)
        .attr('x2', d => (d.target as ConceptNode).x!)
        .attr('y2', d => (d.target as ConceptNode).y!);

      nodes
        .attr('cx', d => d.x!)
        .attr('cy', d => d.y!);

      if (showLabels) {
        g.selectAll('.labels text')
          .attr('x', (d: any) => d.x)
          .attr('y', (d: any) => d.y);
      }
    });

    // Drag functions
    function dragstarted(event: any, d: ConceptNode) {
      if (!event.active) simulation.alphaTarget(0.3).restart();
      d.fx = d.x;
      d.fy = d.y;
    }

    function dragged(event: any, d: ConceptNode) {
      d.fx = event.x;
      d.fy = event.y;
    }

    function dragended(event: any, d: ConceptNode) {
      if (!event.active) simulation.alphaTarget(0);
      d.fx = null;
      d.fy = null;
    }
  };

  const getConceptType = (concept: string): string => {
    const conceptLower = concept.toLowerCase();
    if (conceptLower.includes('lobe') || conceptLower.includes('cortex')) return 'anatomy';
    if (conceptLower.includes('surgery') || conceptLower.includes('approach')) return 'technique';
    if (conceptLower.includes('tumor') || conceptLower.includes('aneurysm')) return 'condition';
    return 'general';
  };

  const getConceptGroup = (concept: string): number => {
    const type = getConceptType(concept);
    switch (type) {
      case 'anatomy': return 1;
      case 'technique': return 2;
      case 'condition': return 3;
      default: return 4;
    }
  };

  const getConceptColor = (concept: string): string => {
    const type = getConceptType(concept);
    switch (type) {
      case 'anatomy': return '#4CAF50';     // Green
      case 'technique': return '#2196F3';   // Blue
      case 'condition': return '#FF9800';   // Orange
      default: return '#9C27B0';           // Purple
    }
  };

  const getStrengthValue = (strength: string): number => {
    switch (strength.toLowerCase()) {
      case 'strong': return 1.0;
      case 'moderate': return 0.7;
      case 'weak': return 0.4;
      default: return 0.5;
    }
  };

  const getLinkColor = (type: string): string => {
    switch (type) {
      case 'anatomical_relation': return '#4CAF50';
      case 'surgical_approach': return '#2196F3';
      case 'disease_progression': return '#FF9800';
      case 'diagnostic_pathway': return '#9C27B0';
      default: return '#999';
    }
  };

  const handleZoomIn = () => {
    if (svgRef.current) {
      d3.select(svgRef.current).transition().call(
        d3.zoom<SVGSVGElement, unknown>().scaleBy as any, 1.5
      );
    }
  };

  const handleZoomOut = () => {
    if (svgRef.current) {
      d3.select(svgRef.current).transition().call(
        d3.zoom<SVGSVGElement, unknown>().scaleBy as any, 0.67
      );
    }
  };

  const handleCenter = () => {
    if (svgRef.current) {
      d3.select(svgRef.current).transition().call(
        d3.zoom<SVGSVGElement, unknown>().transform as any,
        d3.zoomIdentity
      );
      setZoomLevel(1);
    }
  };

  const handleDownload = () => {
    if (svgRef.current) {
      const svgData = new XMLSerializer().serializeToString(svgRef.current);
      const blob = new Blob([svgData], { type: 'image/svg+xml' });
      const url = URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = `knowledge-graph-${centralConcept}.svg`;
      link.click();
      URL.revokeObjectURL(url);
    }
  };

  const renderControls = () => (
    <Card sx={{ mb: 2 }}>
      <CardContent>
        <Grid container spacing={2} alignItems="center">
          <Grid item xs={12} md={4}>
            <TextField
              fullWidth
              label="Central Concept"
              value={centralConcept}
              onChange={(e) => setCentralConcept(e.target.value)}
              onKeyPress={(e) => e.key === 'Enter' && buildConceptGraph(centralConcept)}
            />
          </Grid>
          <Grid item xs={12} md={2}>
            <Button
              variant="contained"
              onClick={() => buildConceptGraph(centralConcept)}
              disabled={loading}
              startIcon={<Search />}
            >
              Build Graph
            </Button>
          </Grid>
          <Grid item xs={12} md={6}>
            <Box sx={{ display: 'flex', gap: 1 }}>
              <Tooltip title="Zoom In">
                <IconButton onClick={handleZoomIn}>
                  <ZoomIn />
                </IconButton>
              </Tooltip>
              <Tooltip title="Zoom Out">
                <IconButton onClick={handleZoomOut}>
                  <ZoomOut />
                </IconButton>
              </Tooltip>
              <Tooltip title="Center Graph">
                <IconButton onClick={handleCenter}>
                  <CenterFocusStrong />
                </IconButton>
              </Tooltip>
              <Tooltip title="Refresh">
                <IconButton onClick={() => buildConceptGraph(centralConcept)}>
                  <Refresh />
                </IconButton>
              </Tooltip>
              <Tooltip title="Settings">
                <IconButton onClick={() => setSettingsOpen(true)}>
                  <Settings />
                </IconButton>
              </Tooltip>
              <Tooltip title="Download SVG">
                <IconButton onClick={handleDownload}>
                  <Download />
                </IconButton>
              </Tooltip>
            </Box>
          </Grid>
        </Grid>
      </CardContent>
    </Card>
  );

  const renderLegend = () => (
    <Card sx={{ mb: 2 }}>
      <CardContent>
        <Typography variant="h6" gutterBottom>
          Legend
        </Typography>
        <Grid container spacing={2}>
          <Grid item xs={6}>
            <Typography variant="subtitle2" gutterBottom>Node Types:</Typography>
            <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1 }}>
              <Chip label="Central Concept" style={{ backgroundColor: '#ff4444', color: 'white' }} size="small" />
              <Chip label="Anatomy" style={{ backgroundColor: '#4CAF50', color: 'white' }} size="small" />
              <Chip label="Technique" style={{ backgroundColor: '#2196F3', color: 'white' }} size="small" />
              <Chip label="Condition" style={{ backgroundColor: '#FF9800', color: 'white' }} size="small" />
            </Box>
          </Grid>
          <Grid item xs={6}>
            <Typography variant="subtitle2" gutterBottom>Relationship Types:</Typography>
            <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1 }}>
              <Chip label="Anatomical" variant="outlined" size="small" />
              <Chip label="Surgical" variant="outlined" size="small" />
              <Chip label="Disease" variant="outlined" size="small" />
              <Chip label="Diagnostic" variant="outlined" size="small" />
            </Box>
          </Grid>
        </Grid>
      </CardContent>
    </Card>
  );

  const renderSettings = () => (
    <Dialog open={settingsOpen} onClose={() => setSettingsOpen(false)} maxWidth="md" fullWidth>
      <DialogTitle>Graph Settings</DialogTitle>
      <DialogContent>
        <Grid container spacing={3}>
          <Grid item xs={12} md={6}>
            <Typography gutterBottom>Node Size</Typography>
            <Slider
              value={nodeSize}
              onChange={(_, value) => setNodeSize(value as number)}
              min={5}
              max={20}
              step={1}
              marks
              valueLabelDisplay="auto"
            />
          </Grid>
          <Grid item xs={12} md={6}>
            <Typography gutterBottom>Link Strength</Typography>
            <Slider
              value={linkStrength}
              onChange={(_, value) => setLinkStrength(value as number)}
              min={0.1}
              max={1.0}
              step={0.1}
              marks
              valueLabelDisplay="auto"
            />
          </Grid>
          <Grid item xs={12} md={6}>
            <Typography gutterBottom>Confidence Filter</Typography>
            <Slider
              value={confidenceFilter}
              onChange={(_, value) => setConfidenceFilter(value as number)}
              min={0.0}
              max={1.0}
              step={0.1}
              marks
              valueLabelDisplay="auto"
            />
          </Grid>
          <Grid item xs={12} md={6}>
            <FormControlLabel
              control={
                <Switch
                  checked={showLabels}
                  onChange={(e) => setShowLabels(e.target.checked)}
                />
              }
              label="Show Labels"
            />
          </Grid>
          <Grid item xs={12}>
            <Typography gutterBottom>Relationship Types</Typography>
            <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1 }}>
              {currentGraph?.metadata.relationshipTypes.map(type => (
                <Chip
                  key={type}
                  label={type}
                  clickable
                  color={selectedTypes.includes(type) ? 'primary' : 'default'}
                  onClick={() => {
                    setSelectedTypes(prev =>
                      prev.includes(type)
                        ? prev.filter(t => t !== type)
                        : [...prev, type]
                    );
                  }}
                />
              ))}
            </Box>
          </Grid>
        </Grid>
      </DialogContent>
      <DialogActions>
        <Button onClick={() => setSettingsOpen(false)}>Close</Button>
      </DialogActions>
    </Dialog>
  );

  const renderNodeDetails = () => (
    <Dialog open={nodeDetailsOpen} onClose={() => setNodeDetailsOpen(false)} maxWidth="sm" fullWidth>
      <DialogTitle>
        {selectedNode ? 'Node Details' : 'Link Details'}
      </DialogTitle>
      <DialogContent>
        {selectedNode && (
          <Box>
            <Typography variant="h6">{selectedNode.name}</Typography>
            <Typography color="textSecondary">Type: {selectedNode.type}</Typography>
            <Typography color="textSecondary">Group: {selectedNode.group}</Typography>
          </Box>
        )}
        {selectedLink && (
          <Box>
            <Typography variant="h6">Relationship</Typography>
            <Typography>Source: {typeof selectedLink.source === 'string' ? selectedLink.source : selectedLink.source.name}</Typography>
            <Typography>Target: {typeof selectedLink.target === 'string' ? selectedLink.target : selectedLink.target.name}</Typography>
            <Typography>Type: {selectedLink.type}</Typography>
            <Typography>Confidence: {(selectedLink.confidence * 100).toFixed(0)}%</Typography>
            <Typography>Description: {selectedLink.description}</Typography>
          </Box>
        )}
      </DialogContent>
      <DialogActions>
        <Button onClick={() => setNodeDetailsOpen(false)}>Close</Button>
      </DialogActions>
    </Dialog>
  );

  const renderStatistics = () => (
    <Card>
      <CardContent>
        <Typography variant="h6" gutterBottom>
          Graph Statistics
        </Typography>
        {currentGraph && (
          <Grid container spacing={2}>
            <Grid item xs={6}>
              <Typography variant="body2">
                Total Concepts: {currentGraph.metadata.totalConcepts}
              </Typography>
            </Grid>
            <Grid item xs={6}>
              <Typography variant="body2">
                Relationships: {currentGraph.links.length}
              </Typography>
            </Grid>
            <Grid item xs={6}>
              <Typography variant="body2">
                Zoom Level: {(zoomLevel * 100).toFixed(0)}%
              </Typography>
            </Grid>
            <Grid item xs={6}>
              <Typography variant="body2">
                Relationship Types: {currentGraph.metadata.relationshipTypes.length}
              </Typography>
            </Grid>
          </Grid>
        )}
      </CardContent>
    </Card>
  );

  return (
    <Box sx={{ p: 3 }}>
      <Typography variant="h4" gutterBottom>
        <AccountTree sx={{ mr: 2, verticalAlign: 'middle' }} />
        Knowledge Graph Visualization
      </Typography>

      {renderControls()}
      {renderLegend()}

      <Grid container spacing={3}>
        <Grid item xs={12} lg={9}>
          <Card>
            <CardContent>
              <Box sx={{ position: 'relative', width: '100%', height: 600, overflow: 'hidden' }}>
                <svg
                  ref={svgRef}
                  width="100%"
                  height="100%"
                  style={{ border: '1px solid #ddd', cursor: 'grab' }}
                />
                {loading && (
                  <Box
                    sx={{
                      position: 'absolute',
                      top: 0,
                      left: 0,
                      right: 0,
                      bottom: 0,
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      backgroundColor: 'rgba(255,255,255,0.8)'
                    }}
                  >
                    <Typography>Building knowledge graph...</Typography>
                  </Box>
                )}
              </Box>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} lg={3}>
          {renderStatistics()}
        </Grid>
      </Grid>

      {renderSettings()}
      {renderNodeDetails()}
    </Box>
  );
};

export default KnowledgeGraphVisualization;