/**
 * Main Layout Component for KOO Platform
 * Combines header, sidebar, and main content area
 */

import React, { useState, useEffect } from 'react';
import {
  Box,
  CssBaseline,
  useTheme,
  useMediaQuery,
  Fab,
  Zoom,
  Snackbar,
  Alert
} from '@mui/material';
import { Add as AddIcon, KeyboardArrowUp as ScrollTopIcon } from '@mui/icons-material';
import { useNavigate, useLocation } from 'react-router-dom';

import Header from './Header';
import Sidebar from './Sidebar';
import { useAuth } from '../../contexts/AuthContext';

interface LayoutProps {
  children: React.ReactNode;
}

const Layout: React.FC<LayoutProps> = ({ children }) => {
  const theme = useTheme();
  const navigate = useNavigate();
  const location = useLocation();
  const { user } = useAuth();

  // Responsive sidebar management
  const isMobile = useMediaQuery(theme.breakpoints.down('md'));
  const [mobileOpen, setMobileOpen] = useState(false);
  const [showScrollTop, setShowScrollTop] = useState(false);
  const [notification, setNotification] = useState<{
    open: boolean;
    message: string;
    severity: 'success' | 'error' | 'warning' | 'info';
  }>({
    open: false,
    message: '',
    severity: 'info'
  });

  // Handle sidebar toggle
  const handleDrawerToggle = () => {
    setMobileOpen(!mobileOpen);
  };

  // Close mobile sidebar on route change
  useEffect(() => {
    if (isMobile) {
      setMobileOpen(false);
    }
  }, [location.pathname, isMobile]);

  // Scroll to top functionality
  useEffect(() => {
    const handleScroll = () => {
      setShowScrollTop(window.scrollY > 300);
    };

    window.addEventListener('scroll', handleScroll);
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  const scrollToTop = () => {
    window.scrollTo({
      top: 0,
      behavior: 'smooth'
    });
  };

  // Quick action - New Chapter
  const handleQuickAction = () => {
    navigate('/chapters/new');
  };

  // Show welcome notification for new users
  useEffect(() => {
    const hasSeenWelcome = localStorage.getItem('koo_welcome_seen');
    if (!hasSeenWelcome) {
      setNotification({
        open: true,
        message: `Welcome to KOO Platform, ${user.full_name}! Start by creating your first chapter.`,
        severity: 'info'
      });
      localStorage.setItem('koo_welcome_seen', 'true');
    }
  }, [user.full_name]);

  const handleNotificationClose = () => {
    setNotification({ ...notification, open: false });
  };

  return (
    <Box sx={{ display: 'flex', minHeight: '100vh' }}>
      <CssBaseline />

      {/* Header */}
      <Header
        onMenuToggle={handleDrawerToggle}
        isMobileMenuOpen={mobileOpen}
      />

      {/* Sidebar */}
      <Sidebar
        open={isMobile ? mobileOpen : true}
        onClose={handleDrawerToggle}
        variant={isMobile ? 'temporary' : 'persistent'}
      />

      {/* Main content area */}
      <Box
        component="main"
        sx={{
          flexGrow: 1,
          bgcolor: 'background.default',
          minHeight: '100vh',
          display: 'flex',
          flexDirection: 'column',
          transition: theme.transitions.create(['margin'], {
            easing: theme.transitions.easing.sharp,
            duration: theme.transitions.duration.leavingScreen,
          }),
          ...(isMobile ? {} : {
            marginLeft: 0,
            transition: theme.transitions.create(['margin'], {
              easing: theme.transitions.easing.easeOut,
              duration: theme.transitions.duration.enteringScreen,
            }),
          }),
        }}
      >
        {/* Toolbar spacing */}
        <Box sx={{ height: 64 }} />

        {/* Page content */}
        <Box
          sx={{
            flexGrow: 1,
            p: { xs: 2, sm: 3 },
            maxWidth: '100%',
            overflow: 'hidden'
          }}
        >
          {children}
        </Box>

        {/* Footer */}
        <Box
          component="footer"
          sx={{
            py: 2,
            px: 3,
            mt: 'auto',
            backgroundColor: 'background.paper',
            borderTop: '1px solid',
            borderColor: 'divider',
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center',
            flexWrap: 'wrap',
            gap: 2
          }}
        >
          <Box>
            <Box component="span" sx={{ color: 'text.secondary', fontSize: '0.875rem' }}>
              KOO Platform - Personal Medical Knowledge Management
            </Box>
          </Box>

          <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
            <Box component="span" sx={{ color: 'text.secondary', fontSize: '0.875rem' }}>
              Version 2.0.0
            </Box>
            <Box
              sx={{
                width: 8,
                height: 8,
                borderRadius: '50%',
                backgroundColor: 'success.main',
                animation: 'pulse 2s infinite'
              }}
              title="System Status: Online"
            />
          </Box>
        </Box>
      </Box>

      {/* Floating Action Button - Quick Chapter Creation */}
      {!isMobile && (
        <Zoom in={true}>
          <Fab
            color="primary"
            aria-label="new chapter"
            onClick={handleQuickAction}
            sx={{
              position: 'fixed',
              bottom: 24,
              right: 24,
              zIndex: theme.zIndex.speedDial
            }}
          >
            <AddIcon />
          </Fab>
        </Zoom>
      )}

      {/* Scroll to Top Button */}
      <Zoom in={showScrollTop}>
        <Fab
          color="secondary"
          size="small"
          aria-label="scroll to top"
          onClick={scrollToTop}
          sx={{
            position: 'fixed',
            bottom: isMobile ? 24 : 96,
            right: 24,
            zIndex: theme.zIndex.speedDial - 1
          }}
        >
          <ScrollTopIcon />
        </Fab>
      </Zoom>

      {/* Global Notification Snackbar */}
      <Snackbar
        open={notification.open}
        autoHideDuration={6000}
        onClose={handleNotificationClose}
        anchorOrigin={{ vertical: 'bottom', horizontal: 'left' }}
      >
        <Alert
          onClose={handleNotificationClose}
          severity={notification.severity}
          variant="filled"
          sx={{ width: '100%' }}
        >
          {notification.message}
        </Alert>
      </Snackbar>

      {/* Global Styles */}
      <style>
        {`
          @keyframes pulse {
            0% { opacity: 1; }
            50% { opacity: 0.5; }
            100% { opacity: 1; }
          }
        `}
      </style>
    </Box>
  );
};

export default Layout;