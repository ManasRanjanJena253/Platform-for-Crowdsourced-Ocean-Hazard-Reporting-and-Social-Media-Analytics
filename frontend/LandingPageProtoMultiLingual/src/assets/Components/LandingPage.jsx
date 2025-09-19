import { Routes, Route, useLocation } from "react-router-dom";
import SuperInteractiveAbout from "../../landingPageComponents/AboutSection";
import ContactSection from "../../landingPageComponents/ContactSection";
import HomeSection from "../../landingPageComponents/HomeSection";
import Navbar from "../../landingPageComponents/Navbar";
import InteractiveServicesSection from "../../landingPageComponents/ServicesSection";
import AuthForm from "../../AuthPage/AuthPage";
import { useTranslation } from 'react-i18next';


function MainContent({ t }) {
     const location = useLocation();

  // 3. Create an array of paths where the navbar should be hidden
  const hideNavbarOnPaths = ['/signin', '/signup'];

  // 4. Check if the current path is in our array
  const showNavbar = !hideNavbarOnPaths.includes(location.pathname);
  return (
    <>
      <Routes>
          <Route path="/signin" element={<AuthForm />} />
          <Route path="/signup" element={<AuthForm />} />
        </Routes>
      
      {showNavbar && <HomeSection t={t} />}

      {showNavbar && <SuperInteractiveAbout t={t} />}

      {showNavbar && <InteractiveServicesSection t={t} />}

      {showNavbar && <ContactSection t={t} />}
    </>
  );
}

export default function LandingPage() {

   const location = useLocation();

  // 3. Create an array of paths where the navbar should be hidden
  const hideNavbarOnPaths = ['/signin', '/signup'];

  // 4. Check if the current path is in our array
  const showNavbar = !hideNavbarOnPaths.includes(location.pathname);


  // for translation
  const { t, i18n } = useTranslation();

  // 3. Define the language change handler here
  const changeLanguage = (lng) => {
    i18n.changeLanguage(lng);
  };

  
  return (
    <div className="scroll-smooth">
      {showNavbar && <Navbar t={t} i18n={i18n} changeLanguage={changeLanguage} />}
        <MainContent t={t} />

    </div>
  );
}
