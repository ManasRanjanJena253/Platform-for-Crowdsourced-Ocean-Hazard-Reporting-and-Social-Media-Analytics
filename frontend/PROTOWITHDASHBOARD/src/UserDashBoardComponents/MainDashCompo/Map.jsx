import React from 'react';
import { motion } from 'framer-motion';
import { Map as MapIcon } from 'lucide-react';
import { GoogleMap, useJsApiLoader, MarkerF, InfoWindowF } from '@react-google-maps/api';


// Assuming mockReports is your imported JSON data
import mockReports from '../../utils/MockData/mockreport.json';


const containerStyle = {
  width: '100%',
  height: '100%'
};

const getMarkerIcon = (urgency) => {
  let color;
  switch (urgency) {
    case 'Critical':
    case 'High':
      color = '#d9534f'; // Red
      break;
    case 'Medium':
      color = '#f0ad4e'; // Orange
      break;
    case 'Low':
      color = '#5bc0de'; // Blue (Info)
      break;
    default:
      color = '#777777'; // Gray
  }
  
  // This check is important because window.google might not be available on the first render
  if (window.google && window.google.maps) {
    return {
      path: window.google.maps.SymbolPath.CIRCLE,
      fillColor: color,
      fillOpacity: 1,
      strokeWeight: 1,
      strokeColor: '#ffffff',
      scale: 9,
      anchor: new window.google.maps.Point(0, 0),
    };
  }
  return {}; // Return empty object or a default if google maps is not loaded
};

export default function MapComponent() {
  const [selectedReport, setSelectedReport] = React.useState(null);
  // ✨ NEW: State to manage the selected time filter
  const [timeFilter, setTimeFilter] = React.useState('all'); 

  // This part is perfect as it is. It converts timestamp strings to Date objects.
  const processedReports = React.useMemo(() => {
    return mockReports.map(report => ({
      ...report,
      timestamp: new Date(report.timestamp),
    }));
  }, []); 

  // ✨ NEW: Memoized filtering logic.
  // This creates a new 'filteredReports' array based on the 'timeFilter' state.
  // It only re-calculates when 'processedReports' or 'timeFilter' changes.
  const filteredReports = React.useMemo(() => {
    const now = new Date();

    if (timeFilter === 'today') {
      // Filter for reports with the same date as today
      return processedReports.filter(report => 
        report.timestamp.toDateString() === now.toDateString()
      );
    }
    
    if (timeFilter === 'this_week') {
      const startOfWeek = new Date(now);
      // Set to the beginning of the week (Sunday)
      startOfWeek.setDate(now.getDate() - now.getDay()); 
      startOfWeek.setHours(0, 0, 0, 0);
      return processedReports.filter(report => report.timestamp >= startOfWeek);
    }

    if (timeFilter === 'this_month') {
      // Set to the first day of the current month
      const startOfMonth = new Date(now.getFullYear(), now.getMonth(), 1);
      return processedReports.filter(report => report.timestamp >= startOfMonth);
    }
    
    // Default case: return all reports
    return processedReports; 
  }, [processedReports, timeFilter]);


  const { isLoaded } = useJsApiLoader({
    id: 'google-map-script',
    googleMapsApiKey: import.meta.env.VITE_GOOGLE_MAPS_API_KEY
  });

  const center = React.useMemo(() => ({
    lat: 15.33,
    lng: 73.88
  }), []);

  if (!isLoaded) {
    return <div>Loading Map...</div>;
  }

  return (
    <motion.div 
      className="bg-white p-4 sm:p-6 rounded-2xl shadow-lg h-[40rem] flex flex-col"
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
    >
      <div className="flex justify-between items-center mb-4">
        <h3 className="text-xl font-bold text-gray-800 flex items-center">
          <MapIcon className="mr-3 text-blue-600" size={24} />
          Reports Near You
        </h3>
        
        {/* ✨ NEW: Dropdown for filtering */}
        <select
          value={timeFilter}
          onChange={(e) => setTimeFilter(e.target.value)}
          className="p-2 border border-gray-300 rounded-md shadow-sm focus:ring-blue-500 focus:border-blue-500"
        >
          <option value="all">All Time</option>
          <option value="today">Today</option>
          <option value="this_week">This Week</option>
          <option value="this_month">This Month</option>
        </select>
      </div>

      <div className="flex-grow rounded-lg overflow-hidden">
        <GoogleMap
          mapContainerStyle={containerStyle}
          center={center}
          zoom={13}
          options={{
            streetViewControl: false,
            mapTypeControl: true,
            fullscreenControl: false,
          }}
        >
          {/* ✨ CHANGE: Map over the new 'filteredReports' array instead of 'processedReports' */}
          {filteredReports.map((report) => (
            <MarkerF
              key={report.report_id}
              position={{ lat: report.location.latitude, lng: report.location.longitude }}
              onClick={() => setSelectedReport(report)}
              icon={getMarkerIcon(report.ai_tags.urgency)}
            />
          ))}

          {selectedReport && (
            <InfoWindowF
              position={{ lat: selectedReport.location.latitude, lng: selectedReport.location.longitude }}
              onCloseClick={() => setSelectedReport(null)}
            >
              <div className="p-1 font-sans">
                <h4 className="font-bold text-md mb-1">{selectedReport.ai_tags.classification}</h4>
                <p><strong>Urgency:</strong> {selectedReport.ai_tags.urgency}</p>
                <p><strong>Source:</strong> {selectedReport.source}</p>
                <p><strong>Time:</strong> {selectedReport.timestamp.toLocaleTimeString()}</p>
                  <a href={selectedReport.report_url} target="_blank" rel="noopener noreferrer" className="text-blue-600 hover:underline">
                    View Details
                  </a>
              </div>
            </InfoWindowF>
          )}
        </GoogleMap>
      </div>
    </motion.div>
  );
}