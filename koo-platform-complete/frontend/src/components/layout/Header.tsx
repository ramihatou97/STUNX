/**
 * Main Header Component for KOO Platform
 * Contains navigation, user info, and quick actions
 */

import React, { useState } from 'react';
import {
  AppBar,
  Toolbar,
  Typography,
  IconButton,
  Button,
  Box,
  Avatar,
  Menu,
  MenuItem,
  Tooltip,
  Badge,
  Chip
} from '@mui/material';
import {
  Menu as MenuIcon,
  Search as SearchIcon,
  Add as AddIcon,
  Notifications as NotificationsIcon,
  Settings as SettingsIcon,
  HealthAndSafety as HealthIcon,
  AccountCircle as AccountIcon
} from '@mui/icons-material';
import { useNavigate } from 'react-router-dom';

import { useAuth } from '../../contexts/AuthContext';

interface HeaderProps {
  onMenuToggle: () => void;
  isMobileMenuOpen: boolean;
}

const Header: React.FC<HeaderProps> = ({ onMenuToggle, isMobileMenuOpen }) => {
  const navigate = useNavigate();
  const { user } = useAuth();
  const [userMenuAnchor, setUserMenuAnchor] = useState<null | HTMLElement>(null);

  const handleUserMenuOpen = (event: React.MouseEvent<HTMLElement>) => {
    setUserMenuAnchor(event.currentTarget);
  };

  const handleUserMenuClose = () => {
    setUserMenuAnchor(null);
  };

  const handleQuickSearch = () => {
    navigate('/search');
  };

  const handleNewChapter = () => {
    navigate('/chapters/new');
  };

  const handleSettings = () => {
    navigate('/settings');
    handleUserMenuClose();
  };

  const isUserMenuOpen = Boolean(userMenuAnchor);

  return (
    <AppBar
      position="fixed"
      sx={{
        zIndex: (theme) => theme.zIndex.drawer + 1,
        backgroundColor: '#1976d2',
        boxShadow: '0 2px 8px rgba(0,0,0,0.1)'
      }}
    >
      <Toolbar sx={{ justifyContent: 'space-between' }}>
        {/* Left section - Menu and Logo */}
        <Box sx={{ display: 'flex', alignItems: 'center' }}>
          <IconButton
            color="inherit"
            aria-label="toggle menu"
            onClick={onMenuToggle}
            edge="start"
            sx={{ mr: 2 }}
          >
            <MenuIcon />
          </IconButton>

          <Box sx={{ display: 'flex', alignItems: 'center', cursor: 'pointer' }} onClick={() => navigate('/')}>
            <HealthIcon sx={{ mr: 1, fontSize: 28 }} />
            <Typography variant="h6" noWrap component="div" sx={{ fontWeight: 600 }}>
              KOO Platform
            </Typography>
            <Chip
              label="Personal"
              size="small"
              sx={{
                ml: 1,
                backgroundColor: 'rgba(255,255,255,0.2)',
                color: 'white',
                fontSize: '0.7rem'
              }}
            />
          </Box>
        </Box>

        {/* Center section - Quick actions */}
        <Box sx={{ display: { xs: 'none', md: 'flex' }, alignItems: 'center', gap: 1 }}>
          <Tooltip title="Quick Search">
            <IconButton color="inherit" onClick={handleQuickSearch}>
              <SearchIcon />
            </IconButton>
          </Tooltip>

          <Tooltip title="New Chapter">
            <Button
              color="inherit"
              startIcon={<AddIcon />}
              onClick={handleNewChapter}
              sx={{
                textTransform: 'none',
                '&:hover': { backgroundColor: 'rgba(255,255,255,0.1)' }
              }}
            >
              New Chapter
            </Button>
          </Tooltip>
        </Box>

        {/* Right section - User info and notifications */}
        <Box sx={{ display: 'flex', alignItems: 'center' }}>
          {/* Notifications */}
          <Tooltip title="Notifications">
            <IconButton color="inherit">
              <Badge badgeContent={0} color="error">
                <NotificationsIcon />
              </Badge>
            </IconButton>
          </Tooltip>

          {/* User Menu */}
          <Tooltip title="User Menu">
            <IconButton
              onClick={handleUserMenuOpen}
              color="inherit"
              sx={{ ml: 1 }}
            >
              <Avatar
                sx={{
                  width: 32,
                  height: 32,
                  backgroundColor: 'rgba(255,255,255,0.2)',
                  fontSize: '0.9rem'
                }}
              >
                {user.full_name.split(' ').map(n => n[0]).join('').toUpperCase()}
              </Avatar>
            </IconButton>
          </Tooltip>

          {/* User info on larger screens */}
          <Box sx={{ display: { xs: 'none', md: 'block' }, ml: 1 }}>
            <Typography variant="body2" sx={{ fontWeight: 500 }}>
              {user.full_name}
            </Typography>
            <Typography variant="caption" sx={{ opacity: 0.8 }}>
              {user.role.replace('_', ' ').toUpperCase()}
            </Typography>
          </Box>
        </Box>

        {/* User Menu Dropdown */}
        <Menu
          anchorEl={userMenuAnchor}
          open={isUserMenuOpen}
          onClose={handleUserMenuClose}
          onClick={handleUserMenuClose}
          PaperProps={{
            elevation: 3,
            sx: {
              overflow: 'visible',
              filter: 'drop-shadow(0px 2px 8px rgba(0,0,0,0.32))',
              mt: 1.5,
              minWidth: 200,
              '& .MuiAvatar-root': {
                width: 32,
                height: 32,
                ml: -0.5,
                mr: 1,
              },
            },
          }}
          transformOrigin={{ horizontal: 'right', vertical: 'top' }}
          anchorOrigin={{ horizontal: 'right', vertical: 'bottom' }}
        >
          <MenuItem sx={{ py: 1.5 }}>
            <Avatar sx={{ mr: 2 }}>
              {user.full_name.split(' ').map(n => n[0]).join('').toUpperCase()}
            </Avatar>
            <Box>
              <Typography variant="body2" sx={{ fontWeight: 500 }}>
                {user.full_name}
              </Typography>
              <Typography variant="caption" color="text.secondary">
                {user.email}
              </Typography>
            </Box>
          </MenuItem>

          <MenuItem onClick={() => navigate('/profile')}>
            <AccountIcon sx={{ mr: 2 }} />
            Profile
          </MenuItem>

          <MenuItem onClick={handleSettings}>
            <SettingsIcon sx={{ mr: 2 }} />
            Settings
          </MenuItem>
        </Menu>
      </Toolbar>
    </AppBar>
  );
};

export default Header;