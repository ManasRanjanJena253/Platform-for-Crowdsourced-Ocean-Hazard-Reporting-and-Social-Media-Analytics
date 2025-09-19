import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { FileUp, Send, AlertCircle, CheckCircle2, X } from 'lucide-react';
import { useParams } from 'react-router-dom';

// Simplified initial state to match the API endpoint's requirements
const initialReportState = {
  location: { latitude: '', longitude: '' },
};

// Blob configurations and styling for the background
const blobConfigurations = [
    { classes: 'w-[650px] h-[650px] top-[-10%] left-[-15%] opacity-80', color: '#cfe0ff' },
    { classes: 'w-[700px] h-[700px] top-[10%] right-[-20%] opacity-70', color: '#f1ddff' },
    { classes: 'w-[550px] h-[550px] bottom-[-20%] left-[10%] opacity-70', color: '#dff7ee' },
    { classes: 'w-[500px] h-[500px] top-[30%] left-[20%] opacity-60', color: '#e9e4ff' },
    { classes: 'w-[600px] h-[600px] top-[5%] left-[50%] opacity-70', color: '#cfe0ff' },
    { classes: 'w-[300px] h-[300px] top-[50%] right-[5%] opacity-80', color: '#f1ddff' },
    { classes: 'w-[350px] h-[350px] bottom-[10%] right-[30%] opacity-70', color: '#dff7ee' },
];

// A simple Toast notification component
const Toast = ({ message, type, onClose }) => {
  const icons = {
    success: <CheckCircle2 className="text-green-500" />,
    error: <AlertCircle className="text-red-500" />,
  };
  const colors = {
    success: 'bg-green-100 border-green-400 text-green-700',
    error: 'bg-red-100 border-red-400 text-red-700',
  };

  useEffect(() => {
    const timer = setTimeout(onClose, 5000);
    return () => clearTimeout(timer);
  }, [onClose]);

  return (
    <motion.div
      initial={{ opacity: 0, y: 50 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, x: 50 }}
      className={`fixed bottom-5 right-5 z-50 p-4 rounded-lg shadow-lg flex items-center border ${colors[type]}`}
    >
      {icons[type]}
      <p className="ml-3 font-medium">{message}</p>
      <button onClick={onClose} className="ml-4 p-1 rounded-full hover:bg-black/10 transition-colors">
        <X size={18} />
      </button>
    </motion.div>
  );
};


export default function NewReportForm() {
  const { userId } = useParams();
  const [reportData, setReportData] = useState(initialReportState);
  const [selectedFile, setSelectedFile] = useState(null);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [toast, setToast] = useState(null);

  // A single handler to manage updates for all fields, including nested ones
  const handleChange = (e) => {
    const { name, value } = e.target;
    
    // Handle nested state updates for location
    if (name.includes('.')) {
      const [parent, child] = name.split('.');
      setReportData(prevData => ({
        ...prevData,
        [parent]: {
          ...prevData[parent],
          [child]: value,
        },
      }));
    } else {
      setReportData(prevData => ({
        ...prevData,
        [name]: value,
      }));
    }
  };

  const handleFileChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      setSelectedFile(e.target.files[0]);
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();

    if (!selectedFile) {
        setToast({ message: "Please select a media file to upload.", type: 'error' });
        return;
    }

    setIsSubmitting(true);
    setToast(null);

    // Prepare data for API call
    const formData = new FormData();
    formData.append('file', selectedFile);

    
    
    // api
    // use userId from line 56;
    const apiUrl = "xyz"

    try {
        // const response = await fetch(apiUrl, {
        //     method: 'POST',
        //     body: formData,
        //     // Headers are not set manually for multipart/form-data;
        //     // the browser sets it with the correct boundary.
        // });

        // const result = await response.json();

        // if (!response.ok) {
        //     // Handle HTTP errors like 500, 404 etc.
        //     throw new Error(result.detail || `Server responded with ${response.status}`);
        // }
        
        // setToast({ message: `Report submitted! ID: ${result.report_id}`, type: 'success' });
        setToast({ message: `Report submitted! ID: `, type: 'success' });
        setReportData(initialReportState); // Reset form
        setSelectedFile(null);
        document.getElementById('file-upload').value = ''; // Clear file input
        
    } catch (error) {
        console.error("Submission Error:", error);
        setToast({ message: `Submission failed: ${error.message}`, type: 'error' });
    } finally {
        setIsSubmitting(false);
    }
  };

  return (
    <section className="relative h-screen overflow-y-auto bg-gray-50/50 w-full pt-28 pb-12 px-4 sm:px-8">
       <AnimatePresence>
        {toast && <Toast message={toast.message} type={toast.type} onClose={() => setToast(null)} />}
      </AnimatePresence>
      
      {/* Background Layer: Blobs and SVG Wave */}
      <div className="absolute inset-0 -z-10 overflow-hidden">
        {blobConfigurations.map((blob, index) => (
          <div
            key={index}
            className={`absolute rounded-full filter blur-xl ${blob.classes}`}
            style={{
              background: `radial-gradient(circle, ${blob.color} 0%, transparent 90%)`,
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
            fillOpacity="1"
            d="M0,64L48,85.3C96,107,192,149,288,160C384,171,480,149,576,144C672,139,768,149,864,170.7C960,192,1056,224,1152,208C1248,192,1344,128,1392,96L1440,64L1440,320L1392,320C1344,320,1248,320,1152,320C1056,320,960,320,864,320C768,320,672,320,576,320C480,320,384,320,288,320C192,320,96,320,48,320L0,320Z"
          ></path>
        </svg>
      </div>

      <motion.div
        className="bg-white/70 backdrop-blur-md p-4 sm:p-8 rounded-2xl shadow-lg w-full"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5, delay: 0.4 }}
      >
        <div className="max-w-xl mx-auto w-auto">
          <div className="text-center mb-8">
            <h3 className="text-2xl font-bold text-gray-800 flex items-center justify-center">
                <FileUp className="mr-3 text-indigo-600" size={28} />
                Upload Hazard Report
            </h3>
            <p className="text-gray-500 mt-2">Provide location details and upload a media file.</p>
          </div>

          <form onSubmit={handleSubmit} className="space-y-6">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
              {/* Latitude */}
              <div>
                <label htmlFor="latitude" className="block text-sm font-medium text-gray-700 mb-1">Latitude</label>
                <input
                  type="number"
                  id="latitude"
                  name="location.latitude"
                  value={reportData.location.latitude}
                  onChange={handleChange}
                  placeholder="e.g., 15.3071"
                  required
                  step="any"
                  className="w-full px-3 py-2 bg-gray-50 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500 transition"
                />
              </div>

              {/* Longitude */}
              <div>
                <label htmlFor="longitude" className="block text-sm font-medium text-gray-700 mb-1">Longitude</label>
                <input
                  type="number"
                  id="longitude"
                  name="location.longitude"
                  value={reportData.location.longitude}
                  onChange={handleChange}
                  placeholder="e.g., 73.8264"
                  required
                  step="any"
                  className="w-full px-3 py-2 bg-gray-50 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500 transition"
                />
              </div>
            </div>
            
            {/* File Upload */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Report Media</label>
              <div className="mt-1 flex justify-center px-6 pt-5 pb-6 border-2 border-gray-300 border-dashed rounded-md">
                <div className="space-y-1 text-center">
                  <svg className="mx-auto h-12 w-12 text-gray-400" stroke="currentColor" fill="none" viewBox="0 0 48 48" aria-hidden="true">
                    <path d="M28 8H12a4 4 0 00-4 4v20m32-12v8m0 0v8a4 4 0 01-4 4H12a4 4 0 01-4-4v-4m32-4l-3.172-3.172a4 4 0 00-5.656 0L28 28M8 32l9.172-9.172a4 4 0 015.656 0L28 28m0 0l4 4m4-24h8m-4-4v8" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
                  </svg>
                  <div className="flex text-sm text-gray-600">
                    <label htmlFor="file-upload" className="relative cursor-pointer bg-white rounded-md font-medium text-indigo-600 hover:text-indigo-500 focus-within:outline-none focus-within:ring-2 focus-within:ring-offset-2 focus-within:ring-indigo-500">
                      <span>Upload a file</span>
                      <input id="file-upload" name="file-upload" type="file" className="sr-only" onChange={handleFileChange} required />
                    </label>
                    <p className="pl-1">or drag and drop</p>
                  </div>
                  <p className="text-xs text-gray-500">PNG, JPG, GIF up to 10MB</p>
                  {selectedFile && <p className="text-sm text-green-600 pt-2 font-semibold">{selectedFile.name}</p>}
                </div>
              </div>
            </div>

            {/* Submit Button */}
            <div className="pt-4">
              <button
                type="submit"
                disabled={isSubmitting}
                className="w-full flex items-center justify-center px-4 py-2.5 font-bold text-white bg-indigo-600 rounded-lg hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500 disabled:bg-indigo-400 disabled:cursor-not-allowed transition-all duration-300"
              >
                {isSubmitting ? (
                  <>
                    <svg className="animate-spin -ml-1 mr-3 h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                    </svg>
                    Submitting...
                  </>
                ) : (
                  <>
                    <Send size={18} className="mr-2" />
                    Submit Report
                  </>
                )}
              </button>
            </div>
          </form>
        </div>
      </motion.div>
    </section>
  );
}