// my reports

import React from 'react';
import { motion } from 'framer-motion';
import { ClipboardList, ExternalLink, Clock, AlertTriangle, MapPin, Database } from 'lucide-react';
import { format, formatDistanceToNow } from 'date-fns';

// blob configurations and styling from the form component
const blobConfigurations = [
    { classes: 'w-[650px] h-[650px] top-[-10%] left-[-15%] opacity-80', color: '#cfe0ff' },
    { classes: 'w-[700px] h-[700px] top-[10%] right-[-20%] opacity-70', color: '#f1ddff' },
    { classes: 'w-[550px] h-[550px] bottom-[-20%] left-[10%] opacity-70', color: '#dff7ee' },
    { classes: 'w-[500px] h-[500px] top-[30%] left-[20%] opacity-60', color: '#e9e4ff' },
    { classes: 'w-[600px] h-[600px] top-[5%] left-[50%] opacity-70', color: '#cfe0ff' },
    { classes: 'w-[300px] h-[300px] top-[50%] right-[5%] opacity-80', color: '#f1ddff' },
];

export default function MyReports() {
  // Updated mock data with more recent timestamps relative to Sept 18, 2025, 2:35 PM
  const myMockReports = [
    {
      "_id": "670f1a9a7b3e8c4d5f6a1b3d",
      "report_id": "REP-CHD-054",
      "report_url": "https://example.com/chandigarh_report/54",
      "location": { "latitude": 30.6493, "longitude": 76.8131 }, // Zirakpur crossing
      "timestamp": "2025-09-18T14:25:00Z", // 10 mins ago
      "ai_tags": { "classification": "Traffic Violation", "urgency": "Critical" },
      "source": "traffic_camera"
    },
    {
      "_id": "670f1a9a7b3e8c4d5f6a1b3a",
      "report_id": "REP-CHD-051",
      "report_url": "https://example.com/chandigarh_report/51",
      "location": { "latitude": 30.7422, "longitude": 76.8185 }, // Sukhna Lake
      "timestamp": "2025-09-18T12:15:00Z", // Approx. 2 hours ago
      "ai_tags": { "classification": "Unattended Object", "urgency": "Medium" },
      "source": "citizen_report"
    },
    {
      "_id": "670f1a9a7b3e8c4d5f6a1b3b",
      "report_id": "REP-CHD-052",
      "report_url": "https://example.com/chandigarh_report/52",
      "location": { "latitude": 30.7176, "longitude": 76.8041 }, // Industrial Area
      "timestamp": "2025-09-18T09:30:00Z", // Approx. 5 hours ago
      "ai_tags": { "classification": "Suspicious Activity", "urgency": "High" },
      "source": "cctv_feed"
    },
    {
      "_id": "670f1a9a7b3e8c4d5f6a1b3c",
      "report_id": "REP-CHD-053",
      "report_url": "https://example.com/chandigarh_report/53",
      "location": { "latitude": 30.7602, "longitude": 76.7633 }, // Near PU
      "timestamp": "2025-09-17T20:00:00Z", // Yesterday
      "ai_tags": { "classification": "Noise Complaint", "urgency": "Low" },
      "source": "citizen_report"
    },
  ];

  const getUrgencyStyles = (urgency) => {
    switch (urgency) {
      case 'Critical':
        return { bg: 'bg-red-50', text: 'text-red-700', border: 'border-red-200', ring: 'focus:ring-red-500' };
      case 'High':
        return { bg: 'bg-orange-50', text: 'text-orange-700', border: 'border-orange-200', ring: 'focus:ring-orange-500' };
      case 'Medium':
        return { bg: 'bg-yellow-50', text: 'text-yellow-700', border: 'border-yellow-200', ring: 'focus:ring-yellow-500' };
      case 'Low':
        return { bg: 'bg-blue-50', text: 'text-blue-700', border: 'border-blue-200', ring: 'focus:ring-blue-500' };
      default:
        return { bg: 'bg-gray-50', text: 'text-gray-700', border: 'border-gray-200', ring: 'focus:ring-gray-500' };
    }
  };

  return (
    <section className="relative h-screen overflow-y-auto bg-gray-50/50 w-full pt-28 pb-12 px-4 sm:px-8">
       {/* Background Layer: Blobs and SVG Wave */}
      <div className="absolute inset-0 -z-10 overflow-hidden">
        {blobConfigurations.map((blob, index) => (
          <div
            key={index}
            className={`absolute rounded-full filter blur-2xl ${blob.classes}`}
            style={{
              background: `radial-gradient(circle, ${blob.color} 0%, transparent 80%)`,
            }}
          />
        ))}
        <div
          className="absolute left-1/2 top-1/2 h-[520px] w-[520px] -translate-x-1/2 -translate-y-1/2 rounded-full opacity-20"
          style={{
            background: "radial-gradient(circle, rgba(0,0,0,0.03) 0%, transparent 60%)",
          }}
        />
        <svg
          className="absolute bottom-0 left-0 w-full"
          viewBox="0 0 1440 320"
          xmlns="http://www.w3.org/2000/svg"
          preserveAspectRatio="none"
        >
          <path
            fill="#e9e4ff"
            fillOpacity="0.8"
            d="M0,64L48,85.3C96,107,192,149,288,160C384,171,480,149,576,144C672,139,768,149,864,170.7C960,192,1056,224,1152,208C1248,192,1344,128,1392,96L1440,64L1440,320L1392,320C1344,320,1248,320,1152,320C1056,320,960,320,864,320C768,320,672,320,576,320C480,320,384,320,288,320C192,320,96,320,48,320L0,320Z"
          ></path>
        </svg>
      </div>

      <motion.div
        className="max-w-7xl mx-auto w-auto"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
      >
        <div className="text-center mb-12">
            <h3 className="text-3xl font-bold text-gray-800 flex items-center justify-center">
                <ClipboardList className="mr-3 text-indigo-600" size={32} />
                My Submitted Reports
            </h3>
            <p className="text-gray-500 mt-2 max-w-2xl mx-auto">
                Here is a list of your recently filed reports. You can track their status and view details below.
            </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 lg:gap-8">
          {myMockReports
           .sort((a, b) => new Date(b.timestamp) - new Date(a.timestamp))
           .map((report) => {
            const urgency = getUrgencyStyles(report.ai_tags.urgency);
            return (
              <motion.div
                key={report._id}
                className={`bg-white/80 backdrop-blur-sm border ${urgency.border} p-5 rounded-xl shadow-md hover:shadow-xl transition-shadow duration-300 flex flex-col`}
                whileHover={{ y: -5 }}
              >
                <div className="flex justify-between items-start mb-4">
                    <div>
                        <h4 className="font-bold text-gray-800 text-lg">{report.ai_tags.classification}</h4>
                        <p className="text-xs text-gray-500 mt-1 font-mono">{report.report_id}</p>
                    </div>
                    <span className={`text-xs font-bold px-3 py-1 rounded-full ${urgency.bg} ${urgency.text}`}>
                        {report.ai_tags.urgency}
                    </span>
                </div>

                <div className="space-y-3 text-sm text-gray-600 flex-grow">
                    <div className="flex items-center">
                        <Database size={14} className="mr-3 text-gray-400" />
                        Source: <strong className="ml-1 text-gray-700">{report.source.replace(/_/g, ' ')}</strong>
                    </div>
                    <div className="flex items-center">
                        <MapPin size={14} className="mr-3 text-gray-400" />
                        Location: <strong className="ml-1 font-mono text-gray-700">{report.location.latitude}, {report.location.longitude}</strong>
                    </div>
                    <div className="flex items-center">
                        <Clock size={14} className="mr-3 text-gray-400" />
                        Time:{' '}
                        <strong className="ml-1 text-gray-700" title={format(new Date(report.timestamp), 'PPpp')}>
                           {formatDistanceToNow(new Date(report.timestamp), { addSuffix: true })}
                        </strong>
                    </div>
                </div>

                <div className="mt-5 pt-4 border-t border-gray-200">
                    <a 
                        href={report.report_url} 
                        target="_blank" 
                        rel="noopener noreferrer"
                        className={`w-full flex items-center justify-center px-4 py-2 font-bold text-white bg-indigo-600 rounded-lg hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 ${urgency.ring} transition-all duration-300`}
                    >
                        <ExternalLink size={16} className="mr-2" />
                        View Full Report
                    </a>
                </div>
              </motion.div>
            )})}
        </div>
      </motion.div>
    </section>
  );
}

// main dash

// map

// import React, { useState, useEffect } from 'react';
// import { motion } from 'framer-motion';
// import { ClipboardList, ExternalLink, Clock, MapPin, Database, Loader, AlertCircle, Inbox } from 'lucide-react';
// import { format, formatDistanceToNow, parseISO } from 'date-fns';
// import { useParams } from 'react-router-dom';

// // Blob configurations and styling from the form component
// const blobConfigurations = [
//     { classes: 'w-[650px] h-[650px] top-[-10%] left-[-15%] opacity-80', color: '#cfe0ff' },
//     { classes: 'w-[700px] h-[700px] top-[10%] right-[-20%] opacity-70', color: '#f1ddff' },
//     { classes: 'w-[550px] h-[550px] bottom-[-20%] left-[10%] opacity-70', color: '#dff7ee' },
//     { classes: 'w-[500px] h-[500px] top-[30%] left-[20%] opacity-60', color: '#e9e4ff' },
//     { classes: 'w-[600px] h-[600px] top-[5%] left-[50%] opacity-70', color: '#cfe0ff' },
//     { classes: 'w-[300px] h-[300px] top-[50%] right-[5%] opacity-80', color: '#f1ddff' },
// ];

// export default function MyReports() {
//     const { userId } = useParams();
//     const [reports, setReports] = useState([]);
//     const [isLoading, setIsLoading] = useState(true);
//     const [error, setError] = useState(null);

//     useEffect(() => {
//         if (!userId) {
//             setError("User ID is missing from the URL.");
//             setIsLoading(false);
//             return;
//         };

//         const fetchReports = async () => {
//             setIsLoading(true);
//             setError(null);
            
//             // api url
//             // const apiUrl = `/list_reports/${userId}`;

//             try {
//                 // const response = await fetch(apiUrl, {
//                 //     method: 'POST', 
//                 //     headers: {
//                 //         'Content-Type': 'application/json'
//                 //     }
//                 // });
                
//                 // if (!response.ok) {
//                 //     const errorData = await response.json();
//                 //     throw new Error(errorData.detail || `Server error: ${response.status}`);
//                 // }
                
//                 // const data = await response.json();
//                 // Ensure the timestamp is a valid Date object for sorting
//                 // const sortedReports = data.reports_list.sort((a, b) => 
//                     // parseISO(b.timestamp) - parseISO(a.timestamp)
//                 // );
//                 // setReports(sortedReports);

//             } catch (err) {
//                 console.error("Failed to fetch reports:", err);
//                 setError(err.message);
//             } finally {
//                 setIsLoading(false);
//             }
//         };

//         fetchReports();
//     }, [userId]); // Re-run the effect if userId changes

//     const getUrgencyStyles = (urgency) => {
//         switch (urgency) {
//             case 'Critical':
//                 return { bg: 'bg-red-50', text: 'text-red-700', border: 'border-red-200', ring: 'focus:ring-red-500' };
//             case 'High':
//             case 'high':
//                 return { bg: 'bg-orange-50', text: 'text-orange-700', border: 'border-orange-200', ring: 'focus:ring-orange-500' };
//             case 'Medium':
//                 return { bg: 'bg-yellow-50', text: 'text-yellow-700', border: 'border-yellow-200', ring: 'focus:ring-yellow-500' };
//             case 'Low':
//             case 'low':
//                 return { bg: 'bg-blue-50', text: 'text-blue-700', border: 'border-blue-200', ring: 'focus:ring-blue-500' };
//             default:
//                 return { bg: 'bg-gray-50', text: 'text-gray-700', border: 'border-gray-200', ring: 'focus:ring-gray-500' };
//         }
//     };

//     const renderContent = () => {
//         if (isLoading) {
//             return (
//                 <div className="flex flex-col items-center justify-center text-center text-gray-500 h-64">
//                     <Loader className="animate-spin text-indigo-500 mb-4" size={48} />
//                     <p className="font-semibold text-lg">Loading Your Reports...</p>
//                     <p>Please wait a moment.</p>
//                 </div>
//             );
//         }

//         if (error) {
//             return (
//                 <div className="flex flex-col items-center justify-center text-center text-red-600 bg-red-50 p-8 rounded-lg h-64">
//                     <AlertCircle className="mb-4" size={48} />
//                     <p className="font-bold text-lg">Failed to Load Reports</p>
//                     <p className="font-mono text-sm bg-red-100 p-2 rounded mt-2">{error}</p>
//                 </div>
//             );
//         }

//         if (reports.length === 0) {
//             return (
//                 <div className="flex flex-col items-center justify-center text-center text-gray-500 h-64">
//                     <Inbox className="text-gray-400 mb-4" size={48} />
//                     <p className="font-semibold text-lg">No Reports Found</p>
//                     <p>You have not submitted any reports yet.</p>
//                 </div>
//             );
//         }

//         return (
//             <div className="grid grid-cols-1 md:grid-cols-2 gap-6 lg:gap-8">
//                 {reports.map((report) => {
//                     const urgency = getUrgencyStyles(report.ai_tags.urgency);
//                     return (
//                         <motion.div
//                             key={report._id}
//                             className={`bg-white/80 backdrop-blur-sm border ${urgency.border} p-5 rounded-xl shadow-md hover:shadow-xl transition-shadow duration-300 flex flex-col`}
//                             whileHover={{ y: -5 }}
//                             initial={{ opacity: 0, y: 20 }}
//                             animate={{ opacity: 1, y: 0 }}
//                             transition={{ duration: 0.3 }}
//                         >
//                             <div className="flex justify-between items-start mb-4">
//                                 <div>
//                                     <h4 className="font-bold text-gray-800 text-lg capitalize">{report.ai_tags.classification || "N/A"}</h4>
//                                     <p className="text-xs text-gray-500 mt-1 font-mono">{report.report_id}</p>
//                                 </div>
//                                 <span className={`text-xs font-bold px-3 py-1 rounded-full capitalize ${urgency.bg} ${urgency.text}`}>
//                                     {report.ai_tags.urgency}
//                                 </span>
//                             </div>
//                             <div className="space-y-3 text-sm text-gray-600 flex-grow">
//                                 <div className="flex items-center">
//                                     <Database size={14} className="mr-3 text-gray-400" />
//                                     Source: <strong className="ml-1 text-gray-700 capitalize">{report.source.replace(/_/g, ' ')}</strong>
//                                 </div>
//                                 <div className="flex items-center">
//                                     <MapPin size={14} className="mr-3 text-gray-400" />
//                                     Location: <strong className="ml-1 font-mono text-gray-700">{report.location.latitude}, {report.location.longitude}</strong>
//                                 </div>
//                                 <div className="flex items-center">
//                                     <Clock size={14} className="mr-3 text-gray-400" />
//                                     Time:{' '}
//                                     <strong className="ml-1 text-gray-700" title={format(parseISO(report.timestamp), 'PPpp')}>
//                                        {formatDistanceToNow(parseISO(report.timestamp), { addSuffix: true })}
//                                     </strong>
//                                 </div>
//                             </div>
//                             <div className="mt-5 pt-4 border-t border-gray-200">
//                                 <a
//                                     href={report.report_url}
//                                     target="_blank"
//                                     rel="noopener noreferrer"
//                                     className={`w-full flex items-center justify-center px-4 py-2 font-bold text-white bg-indigo-600 rounded-lg hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 ${urgency.ring} transition-all duration-300`}
//                                 >
//                                     <ExternalLink size={16} className="mr-2" />
//                                     View Full Report
//                                 </a>
//                             </div>
//                         </motion.div>
//                     )
//                 })}
//             </div>
//         );
//     }

//     return (
//         <section className="relative h-screen overflow-y-auto bg-gray-50/50 w-full pt-28 pb-12 px-4 sm:px-8">
//             <div className="absolute inset-0 -z-10 overflow-hidden">
//                 {blobConfigurations.map((blob, index) => (
//                     <div
//                         key={index}
//                         className={`absolute rounded-full filter blur-2xl ${blob.classes}`}
//                         style={{ background: `radial-gradient(circle, ${blob.color} 0%, transparent 80%)` }}
//                     />
//                 ))}
//                 <div
//                     className="absolute left-1/2 top-1/2 h-[520px] w-[520px] -translate-x-1/2 -translate-y-1/2 rounded-full opacity-20"
//                     style={{ background: "radial-gradient(circle, rgba(0,0,0,0.03) 0%, transparent 60%)" }}
//                 />
//                 <svg className="absolute bottom-0 left-0 w-full" viewBox="0 0 1440 320" xmlns="http://www.w3.org/2000/svg" preserveAspectRatio="none">
//                     <path fill="#e9e4ff" fillOpacity="0.8" d="M0,64L48,85.3C96,107,192,149,288,160C384,171,480,149,576,144C672,139,768,149,864,170.7C960,192,1056,224,1152,208C1248,192,1344,128,1392,96L1440,64L1440,320L1392,320C1344,320,1248,320,1152,320C1056,320,960,320,864,320C768,320,672,320,576,320C480,320,384,320,288,320C192,320,96,320,48,320L0,320Z"></path>
//                 </svg>
//             </div>

//             <motion.div
//                 className="max-w-7xl mx-auto w-auto"
//                 initial={{ opacity: 0, y: 20 }}
//                 animate={{ opacity: 1, y: 0 }}
//                 transition={{ duration: 0.5 }}
//             >
//                 <div className="text-center mb-12">
//                     <h3 className="text-3xl font-bold text-gray-800 flex items-center justify-center">
//                         <ClipboardList className="mr-3 text-indigo-600" size={32} />
//                         My Submitted Reports
//                     </h3>
//                     <p className="text-gray-500 mt-2 max-w-2xl mx-auto">
//                         Here is a list of your recently filed reports. You can track their status and view details below.
//                     </p>
//                 </div>
//                 {renderContent()}
//             </motion.div>
//         </section>
//     );
// }
