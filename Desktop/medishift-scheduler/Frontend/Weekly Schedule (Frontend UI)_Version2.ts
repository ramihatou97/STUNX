import React from 'react';
import { WeeklySchedule } from '../../../../shared/types'; // Adjust path
import { Sun, Moon, Briefcase, Stethoscope, Phone, UserPlus } from 'lucide-react';

interface WeeklySchedulerUIProps {
  schedule: WeeklySchedule;
  residents: { id: string; name: string; }[];
}

const ActivityBadge = ({ activity }: { activity: any }) => {
    let Icon, bgColor, textColor, title;
    
    switch(activity.type || activity.assignmentType) { // Accommodate different assignment types
        case 'OR':
            Icon = Briefcase; bgColor = 'bg-blue-100'; textColor = 'text-blue-800'; title = activity.caseType || 'OR';
            break;
        case 'Clinic':
            Icon = Stethoscope; bgColor = 'bg-green-100'; textColor = 'text-green-800'; title = activity.clinicType || 'Clinic';
            break;
        case 'Day': case 'Weekend': case '24h':
            Icon = Sun; bgColor = 'bg-red-100'; textColor = 'text-red-800'; title = `${activity.type} Call`;
            break;
        case 'Night':
            Icon = Moon; bgColor = 'bg-indigo-100'; textColor = 'text-indigo-800'; title = 'Night Call';
            break;
        case 'Float':
            Icon = UserPlus; bgColor = 'bg-yellow-100'; textColor = 'text-yellow-800'; title = 'Float';
            break;
        case 'Pager':
            Icon = Phone; bgColor = 'bg-purple-100'; textColor = 'text-purple-800'; title = 'Pager';
            break;
        default:
            return null;
    }

    return (
        <div className={`flex items-center p-1.5 rounded-lg ${bgColor} ${textColor}`}>
            <Icon className="h-4 w-4 mr-2 flex-shrink-0" />
            <span className="text-xs font-semibold truncate">{title}</span>
        </div>
    );
};

export const WeeklySchedulerUI: React.FC<WeeklySchedulerUIProps> = ({ schedule, residents }) => {
  if (!schedule) {
    return <div>Loading weekly schedule...</div>;
  }

  return (
    <div className="bg-white p-6 rounded-xl shadow-lg">
      <h2 className="text-2xl font-bold mb-4">Weekly Clinical Schedule</h2>
      <div className="overflow-x-auto border border-gray-200 rounded-lg">
        <table className="min-w-full divide-y divide-gray-200">
          <thead className="bg-gray-50">
            <tr>
              <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider sticky left-0 bg-gray-50 z-10">Resident</th>
              {schedule.days.map((day, index) => (
                <th key={index} className="px-4 py-3 text-center text-xs font-medium text-gray-500 uppercase tracking-wider">
                  {day.date.toDate().toLocaleDateString('en-US', { weekday: 'short' })}
                  <br/>
                  {day.date.toDate().toLocaleDateString('en-US', { month: 'short', day: 'numeric' })}
                </th>
              ))}
            </tr>
          </thead>
          <tbody className="bg-white divide-y divide-gray-200">
            {residents.map(resident => (
              <tr key={resident.id}>
                <td className="px-4 py-3 whitespace-nowrap sticky left-0 bg-white z-10">
                  <div className="text-sm font-medium text-gray-900">{resident.name}</div>
                </td>
                {schedule.days.map((day, index) => {
                  const allActivities = [
                      ...(day.assignments.or?.filter(a => a.residentId === resident.id) || []),
                      ...(day.assignments.clinic?.filter(a => a.residentId === resident.id) || []),
                      ...(day.assignments.call?.filter(a => a.residentId === resident.id) || []),
                  ];
                  return (
                    <td key={index} className="px-4 py-3 whitespace-nowrap align-top">
                      <div className="space-y-1">
                        {allActivities.length > 0 ? (
                          allActivities.map((act, idx) => <ActivityBadge key={idx} activity={act} />)
                        ) : (
                          <span className="text-xs text-gray-400">Off / Unassigned</span>
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
    </div>
  );
};

