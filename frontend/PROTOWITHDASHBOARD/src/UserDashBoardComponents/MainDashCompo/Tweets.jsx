import React from 'react';
import { motion } from 'framer-motion';
import { X, User, MessageCircle, Repeat, Heart, BarChart2 } from 'lucide-react';

import mockPosts from '../../utils/MockData/mockPosts.json';

export default function XFeed() {
  return (
    <motion.div 
      className="bg-white p-4 sm:p-6 rounded-2xl shadow-lg flex flex-col h-[40rem] md:h-[40rem] lg:h-[44rem] font-sans"
      initial={{ opacity: 0, x: 20 }}
      animate={{ opacity: 1, x: 0 }}
      transition={{ duration: 0.5, delay: 0.4 }}
    >
      <h3 className="text-xl font-bold text-gray-800 mb-4 flex items-center flex-shrink-0">
        <X className="mr-3 text-black" size={22} />
        Social Media Activity
      </h3>
      {/* Added custom scrollbar styles for a more modern look */}
      <div className="space-y-1 overflow-y-auto flex-grow -mr-4 pr-4 
                      [&::-webkit-scrollbar]:w-2 
                      [&::-webkit-scrollbar-track]:bg-gray-100 
                      [&::-webkit-scrollbar-thumb]:bg-gray-300 
                      dark:[&::-webkit-scrollbar-track]:bg-slate-700 
                      dark:[&::-webkit-scrollbar-thumb]:bg-slate-500">
        {mockPosts.map((post, index) => (
          // ✨ TWEET CONTAINER (Hover effect removed) ✨
          <motion.div 
            key={index} 
            className="flex items-start space-x-3 p-3 border-b border-gray-100 last:border-b-0 rounded-lg"
            // The whileHover prop and cursor-pointer class have been removed
          >
            {/* User Avatar */}
            <div className="bg-gray-200 rounded-full p-2 flex-shrink-0 mt-1">
              <User size={20} className="text-gray-600" />
            </div>

            {/* Tweet Content */}
            <div className="w-full">
              {/* User Info */}
              <div className="flex items-center space-x-1">
                <p className="font-bold text-gray-900 text-sm hover:underline cursor-pointer">{post.name}</p>
                <p className="text-gray-500 text-sm">@{post.handle}</p>
                <span className="text-gray-500">·</span>
                <p className="text-gray-500 text-sm hover:underline cursor-pointer">{post.timestamp}</p>
              </div>

              {/* Tweet Text */}
              <p className="text-gray-800 text-sm leading-snug my-1">{post.text}</p>
              
              {/* Action Icons */}
              <div className="flex justify-between items-center mt-3 max-w-sm text-gray-500">
                  <div className="flex items-center space-x-1 group cursor-pointer">
                      <MessageCircle size={18} className="group-hover:text-blue-500"/>
                      <span className="text-xs group-hover:text-blue-500">{post.replies}</span>
                  </div>
                  <div className="flex items-center space-x-1 group cursor-pointer">
                      <Repeat size={18} className="group-hover:text-green-500"/>
                      <span className="text-xs group-hover:text-green-500">{post.retweets}</span>
                  </div>
                  <div className="flex items-center space-x-1 group cursor-pointer">
                      <Heart size={18} className="group-hover:text-pink-500"/>
                      <span className="text-xs group-hover:text-pink-500">{post.likes}</span>
                  </div>
                   <div className="flex items-center space-x-1 group cursor-pointer">
                      <BarChart2 size={18} className="group-hover:text-blue-500"/>
                      <span className="text-xs group-hover:text-blue-500">{post.views}</span>
                  </div>
              </div>
            </div>
          </motion.div>
        ))}
      </div>
    </motion.div>
  );
}
