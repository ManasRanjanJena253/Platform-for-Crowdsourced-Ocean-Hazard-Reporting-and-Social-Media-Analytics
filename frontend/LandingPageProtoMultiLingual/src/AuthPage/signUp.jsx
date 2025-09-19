// src/components/SignUpForm.jsx

import React from 'react';
import GoogleIcon from './GoogleIcon';

export default function SignUpForm({ isActive, onSignInClick }) {
  return (
    <div className={`
      absolute top-0 left-0 h-full w-full md:w-1/2 flex items-center justify-center
      transition-all duration-700 ease-in-out
      ${isActive
        ? 'opacity-100 z-50 md:translate-x-full'
        : 'opacity-0 z-10 pointer-events-none md:translate-x-0'
      }
    `}>
      <form className="flex flex-col items-center justify-center px-10 w-full text-center bg-white h-full">
        <h1 className="text-3xl font-bold mb-3">Create Account</h1>
        <a href="#" className="flex items-center justify-center border border-gray-300 rounded-lg h-10 w-10 my-3 hover:bg-gray-100">
          <GoogleIcon />
        </a>
        <span className="text-xs mb-3">Create a User ID for Registration</span>
        <input type="text" placeholder="Name" className="bg-gray-100 border-none rounded-full py-3 px-4 my-2 w-full outline-none shadow-inner" />
        <input type="email" placeholder="Email" className="bg-gray-100 border-none rounded-full py-3 px-4 my-2 w-full outline-none shadow-inner" />
        <input type="password" placeholder="Password" className="bg-gray-100 border-none rounded-full py-3 px-4 my-2 w-full outline-none shadow-inner" />
        <button type="button" className="bg-gradient-to-r from-blue-700 to-purple-700 text-white font-bold uppercase py-3 px-12 rounded-full shadow-lg mt-4 transition-transform duration-100 ease-in hover:scale-105">
          Sign Up
        </button>
        <p className="md:hidden mt-6 text-sm">
          Already have an account?{' '}
          <button type="button" onClick={onSignInClick} className="font-bold text-blue-700 hover:underline">
            Sign In
          </button>
        </p>
      </form>
    </div>
  );
}