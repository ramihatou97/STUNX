/**
 * Sidebar Navigation Component for KOO Platform
 * Contains main navigation menu and quick access
 */

import React from 'react';
import {
  Drawer,
  List,
  ListItem,
  ListItemButton,
  ListItemIcon,
  ListItemText,
  Divider,
  Box,
  Typography,
  Chip,
  Collapse,
  IconButton
} from '@mui/material';
import {
  Dashboard as DashboardIcon,
  Article as ChapterIcon,
  Search as ResearchIcon,
  Upload as FileIcon,
  Analytics as AnalyticsIcon,
  Settings as SettingsIcon,
  Help as HelpIcon,
  ExpandLess,
  ExpandMore,
  Add as AddIcon,
  FolderOpen as FolderIcon,
  History as HistoryIcon,
  Bookmark as BookmarkIcon
} from '@mui/icons-material';
import { useNavigate, useLocation } from 'react-router-dom';
import { useState } from 'react';

interface SidebarProps {
  open: boolean;
  onClose: () => void;
  variant?: 'temporary' | 'persistent' | 'permanent';
}

const drawerWidth = 280;

const Sidebar: React.FC<SidebarProps> = ({ open, onClose, variant = 'temporary' }) => {
  const navigate = useNavigate();
  const location = useLocation();
  const [chaptersExpanded, setChaptersExpanded] = useState(true);
  const [researchExpanded, setResearchExpanded] = useState(false);

  const isActive = (path: string) => location.pathname === path;

  const handleNavigation = (path: string) => {
    navigate(path);
    if (variant === 'temporary') {
      onClose();
    }
  };

  const mainMenuItems = [
    {
      text: 'Dashboard',
      icon: <DashboardIcon />,
      path: '/',
      badge: null
    },
    {
      text: 'Chapters',
      icon: <ChapterIcon />,
      path: '/chapters',
      badge: null,
      expandable: true,
      expanded: chaptersExpanded,
      onToggle: () => setChaptersExpanded(!chaptersExpanded),
      subItems: [
        { text: 'All Chapters', path: '/chapters', icon: <FolderIcon /> },
        { text: 'New Chapter', path: '/chapters/new', icon: <AddIcon /> },
        { text: 'Drafts', path: '/chapters/drafts', icon: <ChapterIcon />, badge: '3' },
        { text: 'Recent', path: '/chapters/recent', icon: <HistoryIcon /> },
        { text: 'Bookmarked', path: '/chapters/bookmarked', icon: <BookmarkIcon /> }
      ]
    },
    {
      text: 'Research',
      icon: <ResearchIcon />,
      path: '/research',
      badge: null,
      expandable: true,
      expanded: researchExpanded,
      onToggle: () => setResearchExpanded(!researchExpanded),
      subItems: [
        { text: 'Search Literature', path: '/research/search', icon: <ResearchIcon /> },
        { text: 'Saved Searches', path: '/research/saved', icon: <BookmarkIcon /> },
        { text: 'Trends', path: '/research/trends', icon: <AnalyticsIcon /> }
      ]
    },
    {
      text: 'Files',
      icon: <FileIcon />,
      path: '/files',
      badge: null
    }
  ];

  const bottomMenuItems = [
    {
      text: 'Analytics',
      icon: <AnalyticsIcon />,
      path: '/analytics',
      badge: null
    },
    {
      text: 'Settings',
      icon: <SettingsIcon />,
      path: '/settings',
      badge: null
    },
    {
      text: 'Help',
      icon: <HelpIcon />,
      path: '/help',
      badge: null
    }
  ];

  const renderMenuItem = (item: any, isSubItem = false) => (
    <React.Fragment key={item.text}>
      <ListItem disablePadding sx={{ pl: isSubItem ? 4 : 0 }}>
        <ListItemButton
          onClick={() => {
            if (item.expandable) {
              item.onToggle();
            } else {
              handleNavigation(item.path);
            }
          }}
          selected={isActive(item.path)}
          sx={{
            borderRadius: 1,
            mx: 1,
            mb: 0.5,
            '&.Mui-selected': {
              backgroundColor: 'primary.main',
              color: 'primary.contrastText',
              '&:hover': {
                backgroundColor: 'primary.dark',
              },
              '& .MuiListItemIcon-root': {
                color: 'primary.contrastText',
              }
            },
            '&:hover': {
              backgroundColor: 'action.hover',
            }
          }}
        >
          <ListItemIcon sx={{ minWidth: 40 }}>
            {item.icon}
          </ListItemIcon>

          <ListItemText
            primary={item.text}
            sx={{
              '& .MuiTypography-root': {
                fontSize: isSubItem ? '0.875rem' : '0.95rem',
                fontWeight: isSubItem ? 400 : 500
              }
            }}
          />

          {item.badge && (
            <Chip
              label={item.badge}
              size="small"
              sx={{
                height: 20,
                fontSize: '0.7rem',
                backgroundColor: 'error.main',
                color: 'error.contrastText'
              }}
            />
          )}

          {item.expandable && (
            <IconButton
              size="small"
              sx={{ ml: 1 }}
              onClick={(e) => {
                e.stopPropagation();
                item.onToggle();
              }}
            >
              {item.expanded ? <ExpandLess /> : <ExpandMore />}
            </IconButton>
          )}
        </ListItemButton>
      </ListItem>

      {item.expandable && (
        <Collapse in={item.expanded} timeout="auto" unmountOnExit>
          <List component="div" disablePadding>
            {item.subItems?.map((subItem: any) => renderMenuItem(subItem, true))}
          </List>
        </Collapse>
      )}
    </React.Fragment>
  );

  const drawerContent = (
    <Box sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
      {/* Header space */}
      <Box sx={{ height: 64 }} />

      {/* Main Navigation */}
      <Box sx={{ flexGrow: 1, overflow: 'auto', py: 1 }}>
        <Box sx={{ px: 2, mb: 2 }}>
          <Typography variant="overline" color="text.secondary" sx={{ fontWeight: 600 }}>
            Main Navigation
          </Typography>
        </Box>

        <List>
          {mainMenuItems.map((item) => renderMenuItem(item))}
        </List>

        <Divider sx={{ my: 2, mx: 2 }} />

        {/* Quick Stats */}
        <Box sx={{ px: 2, mb: 2 }}>
          <Typography variant="overline" color="text.secondary" sx={{ fontWeight: 600 }}>
            Quick Stats
          </Typography>
          <Box sx={{ mt: 1, display: 'flex', flexDirection: 'column', gap: 1 }}>
            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <Typography variant="body2" color="text.secondary">
                Total Chapters
              </Typography>
              <Chip label="12" size="small" variant="outlined" />
            </Box>
            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <Typography variant="body2" color="text.secondary">
                Files Uploaded
              </Typography>
              <Chip label="8" size="small" variant="outlined" />
            </Box>
            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <Typography variant="body2" color="text.secondary">
                Searches Today
              </Typography>
              <Chip label="5" size="small" variant="outlined" />
            </Box>
          </Box>
        </Box>

        <Divider sx={{ my: 2, mx: 2 }} />
      </Box>

      {/* Bottom Navigation */}
      <Box sx={{ py: 1 }}>
        <Box sx={{ px: 2, mb: 1 }}>
          <Typography variant="overline" color="text.secondary" sx={{ fontWeight: 600 }}>
            System
          </Typography>
        </Box>

        <List>
          {bottomMenuItems.map((item) => renderMenuItem(item))}
        </List>

        {/* Version Info */}
        <Box sx={{ px: 2, py: 1 }}>
          <Typography variant="caption" color="text.secondary">
            Version 2.0.0 - Personal Edition
          </Typography>
        </Box>
      </Box>
    </Box>
  );

  return (
    <Drawer
      variant={variant}
      open={open}
      onClose={onClose}
      sx={{
        width: drawerWidth,
        flexShrink: 0,
        '& .MuiDrawer-paper': {
          width: drawerWidth,
          boxSizing: 'border-box',
          backgroundColor: 'background.paper',
          borderRight: '1px solid',
          borderColor: 'divider',
        },
      }}
    >
      {drawerContent}
    </Drawer>
  );
};

export default Sidebar;