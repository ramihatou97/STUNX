import React, { useState, useEffect } from 'react';
import { 
  Calendar, 
  Users, 
  BarChart, 
  CheckCircle, 
  XCircle, 
  ChevronLeft, 
  ChevronRight, 
  Eye,
  Download,
  Settings,
  RefreshCw
} from 'lucide-react';
import { AcademicYear, RotationAssignment, Resident, RotationBlock } from '../../types';
import { useSchedule } from '../../hooks/useSchedule';
import { api } from '../../services/api';
import LoadingSpinner from '../common/LoadingSpinner';
import ErrorMessage from '../common/ErrorMessage';
import toast from 'react-hot-toast';

const getRotationColor = (rotationType: RotationAssignment['rotationType']): string => {
  const colors: Record<RotationAssignment['rotationType'], string> = {
    'CORE_NSX': 'bg-blue-500 hover:bg-blue-600',
    'MANDATORY_OFF_SERVICE': 'bg-red-500 hover:bg-red-600',
    'FLEXIBLE_OFF_SERVICE': 'bg-yellow-500 hover:bg-yellow-600',
    'RESEARCH': 'bg-green-500 hover:bg-green-600',
    'EXAM_LEAVE': 'bg-purple-500 hover:bg-purple-600',
    'HOLIDAY_LEAVE': 'bg-pink-500 hover:bg-pink-600'
  };
  return colors[rotationType] || 'bg-gray-400 hover:bg-gray-500';
};

interface BlockDetailModalProps {
  block: RotationBlock;
  residents: Resident[];
  onClose: () => void;
}

const BlockDetailModal: React.FC<BlockDetailModalProps> = ({ block, residents, onClose }) => {
  return (
    <div className="fixed inset-0 bg-black bg-opacity-60 z-50 flex justify-center items-center p-4">
      <div className="bg-white rounded-xl shadow-2xl p-6 w-full max-w-2xl">
        <h3 className="text-xl font-bold text-gray-800">
          Block {block.blockNumber} Details
        </h3>
        <p className="text-sm text-gray-500 mb-4">
          {block.startDate.toDate().toLocaleDateString('en-US', { dateStyle: 'long' })} - 
          {block.endDate.toDate().toLocaleDateString('en-US', { dateStyle: 'long' })}
        </p>
        <div className="space-y-3 max-h-96 overflow-y-auto pr-2">
          {block.assignments.map(assignment => {
            const resident = residents.find(r => r.id === assignment.residentId);
            return (
              <div key={assignment.residentId} className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
                <div>
                  <p className="font-semibold">
                    {resident?.name} 
                    <span className="text-xs font-normal text-gray-500 ml-2">
                      PGY{resident?.pgyLevel}
                    </span>
                  </p>
                  <p className="text-sm text-gray-600">{assignment.rotationName}</p>
                  {assignment.team && (
                    <span className={`text-xs font-medium ${
                      assignment.team === 'Red' ? 'text-red-600' : 'text-blue-600'
                    }`}>
                      Team {assignment.team}
                    </span>
                  )}
                </div>
                <div className={`text-xs font-bold px-2 py-1 rounded-full text-white ${getRotationColor(assignment.rotationType)}`}>
                  {assignment.rotationType.replace(/_/g, ' ')}
                </div>
              </div>
            );
          })}
        </div>
        <div className="mt-6 flex justify-end gap-3">
          <button 
            onClick={onClose} 
            className="px-4 py-2 bg-gray-200 text-gray-700 rounded-lg hover:bg-gray-300 transition-colors"
          >
            Close
          </button>
        </div>
      </div>
    </div>
  );
};

export const YearlyScheduler: React.FC = () => {
  const { schedule, residents, loading, error, generateSchedule, exportSchedule } = useSchedule();
  const [selectedBlock, setSelectedBlock] = useState<RotationBlock | null>(null);
  const [academicYear, setAcademicYear] = useState('2025-2026');
  const [isGenerating, setIsGenerating] = useState(false);

  const handleGenerateSchedule = async () => {
    setIsGenerating(true);
    try {
      await generateSchedule(academicYear);
      toast.success('Yearly schedule generated successfully!');
    } catch (error: any) {
      toast.error(error.message || 'Failed to generate schedule');
    } finally {
      setIsGenerating(false);
    }
  };

  const handleExportSchedule = () => {
    if (schedule) {
      exportSchedule(schedule, 'yearly-schedule.csv');
      toast.success('Schedule exported successfully!');
    }
  };

  if (loading) return <LoadingSpinner />;
  if (error) return <ErrorMessage message={error} />;
  if (!schedule) return <ErrorMessage message="No schedule data available" />;

  return (
    <div className="p-4 sm:p-6 bg-gray-50 font-sans">
      <div className="bg-white p-6 rounded-xl shadow-lg">
        {/* Header */}
        <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center mb-6">
          <div>
            <h2 className="text-2xl font-bold text-gray-800 flex items-center">
              <Calendar className="mr-3 text-indigo-600"/>
              Yearly Rotation Schedule
            </h2>
            <p className="text-gray-500">Academic Year: {schedule.id}</p>
          </div>
          
          {/* Controls */}
          <div className="flex items-center gap-2 mt-4 sm:mt-0">
            <button 
              onClick={handleGenerateSchedule}
              disabled={isGenerating}
              className="flex items-center px-3 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 disabled:opacity-50"
            >
              {isGenerating ? <LoadingSpinner size="sm" /> : <RefreshCw className="h-4 w-4 mr-1" />}
              Generate
            </button>
            <button 
              onClick={handleExportSchedule}
              className="flex items-center px-3 py-2 bg-gray-200 text-gray-700 rounded-lg hover:bg-gray-300"
            >
              <Download className="h-4 w-4 mr-1" />
              Export
            </button>
            <button className="p-2 rounded-lg hover:bg-gray-100">
              <Settings className="h-5 w-5" />
            </button>
          </div>
        </div>

        {/* Schedule Grid */}
        <div className="overflow-x-auto">
          <div 
            className="grid gap-1" 
            style={{ 
              gridTemplateColumns: `150px repeat(${schedule.blocks.length}, minmax(100px, 1fr))` 
            }}
          >
            {/* Header Row */}
            <div className="font-bold p-2 sticky left-0 bg-white z-10">Resident</div>
            {schedule.blocks.map(block => (
              <div key={block.blockNumber} className="font-bold text-center p-2 bg-gray-100 rounded-t-md">
                B{block.blockNumber}
                <div className="text-xs font-normal text-gray-400">
                  {block.startDate.toDate().toLocaleDateString('en-US', { month: 'short', day: 'numeric' })}
                </div>
              </div>
            ))}

            {/* Resident Rows */}
            {residents.map(resident => (
              <React.Fragment key={resident.id}>
                <div className="font-medium p-2 sticky left-0 bg-white z-10 flex items-center text-sm">
                  {resident.name} 
                  <span className="ml-2 text-xs bg-gray-200 text-gray-600 px-1.5 py-0.5 rounded-full">
                    PGY{resident.pgyLevel}
                  </span>
                </div>
                {schedule.blocks.map(block => {
                  const assignment = block.assignments.find(a => a.residentId === resident.id);
                  return (
                    <div key={block.blockNumber} className="p-1">
                      {assignment ? (
                        <button 
                          onClick={() => setSelectedBlock(block)} 
                          className={`w-full h-full rounded p-2 text-center text-xs font-semibold text-white flex items-center justify-center transition-transform transform hover:scale-105 ${getRotationColor(assignment.rotationType)}`}
                        >
                          {assignment.rotationName.split(' ')[0]}
                          {assignment.team && (
                            <span className={`ml-1 w-3 h-3 rounded-full border-2 border-white ${
                              assignment.team === 'Red' ? 'bg-red-600' : 'bg-blue-300'
                            }`} />
                          )}
                        </button>
                      ) : (
                        <div className="bg-gray-200 w-full h-full rounded" />
                      )}
                    </div>
                  );
                })}
              </React.Fragment>
            ))}
          </div>
        </div>
      </div>
      
      {/* Block Detail Modal */}
      {selectedBlock && (
        <BlockDetailModal 
          block={selectedBlock} 
          residents={residents} 
          onClose={() => setSelectedBlock(null)} 
        />
      )}
    </div>
  );
};