import React from 'react'
import { Routes, Route } from "react-router-dom";
import DashboardNavbar from '../../UserDashBoardComponents/DashNavbar'
import MainDash from '../../UserDashBoardComponents/mainDash'
import NewReportForm from '../../UserDashBoardComponents/NewReport';
import MyReports from '../../UserDashBoardComponents/MyReposts';

const UserDashBoard = () => {
  return (
    <>
    <DashboardNavbar/>
    <Routes>
      <Route path="/dashboard/new-report" element={<NewReportForm />} />
      <Route path="/dashboard" element={<MainDash />} />
      <Route path='/dashboard/my-reports' element={<MyReports/>}/>

    </Routes>
    
    
    </>
  )
}

export default UserDashBoard