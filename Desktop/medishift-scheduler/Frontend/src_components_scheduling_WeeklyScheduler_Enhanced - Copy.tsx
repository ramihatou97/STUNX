import React, { useState, useEffect, useCallback } from 'react';
import { 
  Calendar,
  Briefcase,
  Stethoscope,
  Sun,
  Moon,
  UserPlus,
  Phone,
  RefreshCw,
  Download,
  ChevronLeft,
  ChevronRight,
  Settings,
  AlertTriangle
} from 'lucide-react';
import { 
  WeeklySchedule,
  DailySchedule,
  ORAssignment,
  ClinicAssignment,
  CallAssignment,
  FloatAssignment,
  PagerAssignment,
  Resident
} from '../../types';
import { useWeeklySchedule } from '../../hooks/useWeeklySchedule';
import { api } from '../../services/api';
import { mlService } from '../../services/ml-service';
import toast from 'react-hot-toast';
import LoadingSpinner from '../common/LoadingSpinner';

interface WeeklySchedulerProps {
  initialWeek?: Date;
}

interface ActivityBadgeProps {
  activity: any;
  onClick?: () => void;
}

const ActivityBadge: React.FC<ActivityBadgeProps> = ({ activity, onClick }) => {
  let Icon, bgColor, textColor, title;
  
  switch(activity.type || activity.assignmentType) {
    case 'OR':
      Icon = Briefcase; 
      bgColor = 'bg-blue-100'; 
      textColor = 'text-blue-800'; 
      title = activity.procedureName || 'OR';
      break;
    case 'Clinic':
      Icon = Stethoscope; 
      bgColor = 'bg-green-100'; 
      textColor = 'text-green-800'; 
      title = activity.clinicType || 'Clinic';
      break;
    case 'Day':
    case 'Weekend':
    case '24h':
      Icon = Sun; 
      bgColor = 'bg-red-100'; 
      textColor = 'text-red-800'; 
      title = `${activity.type} Call`;
      break;
    case 'Night':
      Icon = Moon; 
      bgColor = 'bg-indigo-100'; 
      textColor = 'text-indigo-800'; 
      title = 'Night Call';
      break;
    case 'Float':
      Icon = UserPlus; 
      bgColor = 'bg-yellow-100'; 
      textColor = 'text-yellow-800'; 
      title = 'Float';
      break;
    case 'Pager':
      Icon = Phone; 
      bgColor = 'bg-purple-100'; 
      textColor = 'text-purple-800'; 
      title = `${activity.pagerType} Pager`;
      break;
    default:
      return null;
  }

  return (
    <button
      onClick={onClick}
      className={`flex items-center p-1.5 rounded-lg ${bgColor} ${textColor} hover:opacity-80 transition-opacity`}
    >
      <Icon className="h-4 w-4 mr-2 flex-shrink-0" />
      <span className="text-xs font-semibold truncate">{title}</span>
    </button>
  );
};

export const WeeklyScheduler: React.FC<WeeklySchedulerProps> = ({ 
  initialWeek = new Date() 
}) => {
  const [currentWeek, setCurrentWeek] = useState(initialWeek);
  const [isGenerating, setIsGenerating] = useState(false);
  const [selectedDay, setSelectedDay] = useState<DailySchedule | null>(null);
  const [showSettings, setShowSettings] = useState(false);
  const [conflicts, setConflicts] = useState<any[]>([]);
  
  const { 
    schedule, 
    residents, 
    loading, 
    error, 
    refreshSchedule,
    updateAssignment 
  } = useWeeklySchedule(currentWeek);

  // Check for conflicts on schedule load
  useEffect(() => {
    if (schedule) {
      checkForConflicts();
    }
  }, [schedule]);

  const checkForConflicts = async () => {
    if (!schedule) return;
    
    try {
      const anomalies = await mlService.detectAnomalies({
        type: 'weekly',
        schedule: schedule
      });
      
      if (anomalies.conflicts) {
        setConflicts(anomalies.conflicts);
        if (anomalies.conflicts.length > 0) {
          toast.warning(`${anomalies.conflicts.length} scheduling conflicts detected`);
        }
      }
    } catch (error) {
      console.error('Failed to check conflicts:', error);
    }
  };

  const generateSchedule = async () => {
    setIsGenerating(true);
    try {
      const weekStart = getWeekStart(currentWeek);
      
      const result = await api.generateWeeklySchedule({
        weekStartDate: weekStart.toISOString(),
        residents: residents.map(r => r.id),
        useMLOptimization: true,
        includeFloat: true,
        includePager: true
      });
      
      if (result.data.success) {
        toast.success('Weekly schedule generated successfully!');
        await refreshSchedule();
      }
    } catch (error: any) {
      toast.error(error.message || 'Failed to generate schedule');
    } finally {
      setIsGenerating(false);
    }
  };

  const handleAssignmentUpdate = async (
    dayIndex: number,
    assignmentType: string,
    assignmentId: string,
    updates: any
  ) => {
    try {
      await updateAssignment(dayIndex, assignmentType, assignmentId, updates);
      toast.success('Assignment updated');
    } catch (error) {
      toast.error('Failed to update assignment');
    }
  };

  const exportSchedule = async () => {
    if (!schedule) return;
    
    try {
      const blob = await api.exportWeeklySchedule({
        scheduleId: schedule.id,
        format: 'pdf',
        includeDetails: true
      });
      
      const url = window.URL.createObjectURL(blob.data);
      const a = document.createElement('a');
      a.href = url;
      a.download = `weekly_schedule_${schedule.id}.pdf`;
      a.click();
      
      toast.success('Schedule exported successfully!');
    } catch (error) {
      toast.error('Failed to export schedule');
    }
  };

  const getWeekStart = (date: Date): Date => {
    const d = new Date(date);
    const day = d.getDay();
    const diff = d.getDate() - day;
    return new Date(d.setDate(diff));
  };

  const navigateWeek = (direction: 'prev' | 'next') => {
    const newWeek = new Date(currentWeek);
    newWeek.setDate(newWeek.getDate() + (direction === 'next' ? 7 : -7));
    setCurrentWeek(newWeek);
  };

  const renderTimeSlot = (hour: number, daySchedule: DailySchedule) => {
    const timeString = `${hour.toString().padStart(2, '0')}:00`;
    
    // Find assignments for this time slot
    const orAssignments = daySchedule.assignments.or.filter(a => {
      const startHour = parseInt(a.startTime.split(':')[0]);
      const endHour = parseInt(a.endTime.split(':')[0]);
      return hour >= startHour && hour < endHour;
    });
    
    const clinicAssignments = daySchedule.assignments.clinic.filter(a => {
      const startHour = parseInt(a.startTime.split(':')[0]);
      const endHour = parseInt(a.endTime.split(':')[0]);
      return hour >= startHour && hour < endHour;
    });
    
    return (
      <div className="flex gap-1">
        {orAssignments.map((assignment, idx) => (
          <ActivityBadge
            key={`or-${idx}`}
            activity={{ ...assignment, type: 'OR' }}
            onClick={() => handleAssignmentClick('or', assignment)}
          />
        ))}
        {clinicAssignments.map((assignment, idx) => (
          <ActivityBadge
            key={`clinic-${idx}`}
            activity={{ ...assignment, type: 'Clinic' }}
            onClick={() => handleAssignmentClick('clinic', assignment)}
          />
        ))}
      </div>
    );
  };

  const handleAssignmentClick = (type: string, assignment: any) => {
    // Show assignment details modal
    console.log('Assignment clicked:', type, assignment);
  };

  if (loading) return <LoadingSpinner />;
  if (error) return <div className="text-red-500">Error: {error}</div>;
  if (!schedule) return <div>No schedule data available</div>;

  return (
    <div className="p-6 bg-gray-100">
      <div className="bg-white p-6 rounded-xl shadow-lg">
        {/* Header */}
        <div className="flex justify-between items-center mb-6">
          <h1 className="text-2xl font-bold text-gray-800 flex items-center">
            <Calendar className="mr-2 text-indigo-600" />
            Weekly Clinical Schedule
          </h1>
          
          {/* Week Navigation */}
          <div className="flex items-center gap-4">
            <button 
              onClick={() => navigateWeek('prev')}
              className="p-2 hover:bg-gray-100 rounded"
            >
              <ChevronLeft className="h-5 w-5" />
            </button>
            
            <h2 className="text-lg font-semibold">
              Week {schedule.weekNumber}, {schedule.year}
            </h2>
            
            <button 
              onClick={() => navigateWeek('next')}
              className="p-2 hover:bg-gray-100 rounded"
            >
              <ChevronRight className="h-5 w-5" />
            </button>
          </div>
        </div>

        {/* Controls */}
        <div className="flex flex-wrap gap-3 mb-6">
          <button 
            onClick={generateSchedule} 
            disabled={isGenerating}
            className="flex items-center px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 disabled:opacity-50"
          >
            {isGenerating ? <LoadingSpinner size="sm" /> : <RefreshCw className="mr-2 h-4 w-4"/>}
            Generate
          </button>
          
          <button 
            onClick={exportSchedule}
            className="flex items-center px-4 py-2 bg-gray-600 text-white rounded-lg hover:bg-gray-700"
          >
            <Download className="mr-2 h-4 w-4"/>
            Export
          </button>
          
          <button 
            onClick={() => setShowSettings(true)}
            className="flex items-center px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700"
          >
            <Settings className="mr-2 h-4 w-4"/>
            Settings
          </button>
        </div>

        {/* Conflicts Alert */}
        {conflicts.length > 0 && (
          <div className="mb-4 p-3 bg-red-50 border border-red-200 rounded-lg">
            <div className="flex items-center">
              <AlertTriangle className="h-5 w-5 text-red-600 mr-2" />
              <span className="text-sm text-red-800">
                {conflicts.length} scheduling conflict(s) detected
              </span>
            </div>
          </div>
        )}

        {/* Schedule Grid */}
        <div className="overflow-x-auto">
          <table className="min-w-full divide-y divide-gray-200">
            <thead className="bg-gray-50">
              <tr>
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase sticky left-0 bg-gray-50 z-10">
                  Resident
                </th>
                {schedule.days.map((day, index) => (
                  <th key={index} className="px-4 py-3 text-center text-xs font-medium text-gray-500 uppercase">
                    {day.date.toDate().toLocaleDateString('en-US', { weekday: 'short' })}
                    <br/>
                    {day.date.toDate().toLocaleDateString('en-US', { month: 'short', day: 'numeric' })}
                    {day.isHoliday && (
                      <span className="ml-1 text-red-500">🎄</span>
                    )}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody className="bg-white divide-y divide-gray-200">
              {residents.map(resident => (
                <tr key={resident.id}>
                  <td className="px-4 py-3 whitespace-nowrap sticky left-0 bg-white z-10">
                    <div>
                      <div className="text-sm font-medium text-gray-900">
                        {resident.name}
                      </div>
                      <div className="text-xs text-gray-500">
                        PGY-{resident.pgyLevel}
                      </div>
                    </div>
                  </td>
                  {schedule.days.map((day, dayIndex) => {
                    const allActivities = [
                      ...(day.assignments.or?.filter(a => a.residentId === resident.id) || []).map(a => ({...a, type: 'OR'})),
                      ...(day.assignments.clinic?.filter(a => a.residentId === resident.id) || []).map(a => ({...a, type: 'Clinic'})),
                      ...(day.assignments.call?.filter(a => a.residentId === resident.id) || []),
                      ...(day.assignments.float?.filter(a => a.residentId === resident.id) || []).map(a => ({...a, type: 'Float'})),
                      ...(day.assignments.pager?.filter(a => a.residentId === resident.id) || []).map(a => ({...a, type: 'Pager'}))
                    ];
                    
                    return (
                      <td key={dayIndex} className="px-4 py-3 align-top">
                        <div className="space-y-1">
                          {allActivities.length > 0 ? (
                            allActivities.map((activity, idx) => (
                              <ActivityBadge 
                                key={idx} 
                                activity={activity}
                                onClick={() => handleAssignmentClick(activity.type, activity)}
                              />
                            ))
                          ) : (
                            <span className="text-xs text-gray-400">Off</span>
                          )}
                        </div>
                      </td>
                    );
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        {/* Daily Summary */}
        <div className="mt-6 grid grid-cols-7 gap-2">
          {schedule.days.map((day, index) => {
            const totalOR = day.assignments.or.length;
            const totalClinic = day.assignments.clinic.length;
            const totalCall = day.assignments.call.length;
            
            return (
              <div key={index} className="bg-gray-50 p-3 rounded-lg">
                <h4 className="text-xs font-semibold text-gray-600 mb-2">
                  {day.date.toDate().toLocaleDateString('en-US', { weekday: 'short' })}
                </h4>
                <div className="space-y-1 text-xs">
                  <div>OR: {totalOR}</div>
                  <div>Clinic: {totalClinic}</div>
                  <div>Call: {totalCall}</div>
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
};