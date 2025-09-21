import React, { useState } from "react";
import "./style.css";
import { useNavigate } from "react-router-dom";

function SignIn() {
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [passKey, setPassKey] = useState("");
  const baseApi = 'http://localhost:5000/';
  const navigate = useNavigate();
  const handleLogin = async (e) => {
    e.preventDefault();
    try {
      const response = await fetch(`${baseApi}api/auth/login`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ username, password, passKey }),
      });
      const data = await response.json();
      if (response.ok) {
        localStorage.setItem("token", data.token);
        alert("Login successful! " + data.token);
        navigate("/dashboard");
      } else {
        alert('Login failed. Please check your credentials.');
      }
    } catch (error) {
      console.error('Error during login:', error);
    }
  };

  return (
    <form onSubmit={handleLogin}>
      <h1>Sign In</h1>
      <input type="text" placeholder="UserName" value={username} onChange={(e) => setUsername(e.target.value)} />
      <input type="password" placeholder="Password" value={password} onChange={(e) => setPassword(e.target.value)} />
      <input type="password" placeholder="PassKey" value={passKey} onChange={(e) => setPassKey(e.target.value)} />
      <button type="submit">Sign In</button>
    </form>
  );
}

function SignUp() {
  const [name, setName] = useState("");
  const [email, setEmail] = useState("");
  //Remember to add a state for passkey
  const [password, setPassword] = useState("");

  const baseApi = 'http://localhost:5000/';

  const handleSignup = async (e) => {
    e.preventDefault();
    const response = await fetch(`${baseApi}api/signup`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name, email, password }),
    });
    const data = await response.json();
    if (response.ok) {
      alert("Signup successful!");
    } else {
      alert("Signup failed: " + data.message);
    }
  };

  return (
    <form onSubmit={handleSignup}>
      <h1>Create Account</h1>
      <div className="social-icons">
        <a href="#" className="icon"><i className="fa-brands fa-google-plus-g"></i></a>
      </div>
      <span> Registration For Authorised Personel</span>
      <input type="text" placeholder="Name" value={name} onChange={e => setName(e.target.value)} />
      <input type="email" placeholder="Email" value={email} onChange={e => setEmail(e.target.value)} />
      <input type="password" placeholder="Password" value={password} onChange={e => setPassword(e.target.value)} />
      <button type="submit">Sign Up</button>
    </form>
  );
}

export default function App() {
  const [active, setActive] = useState(false);

  return (
    <div className={`container ${active ? "active" : ""}`} id="container">
      {/* Sign Up */}
      <div className="form-container sign-up">
        <SignUp />
      </div>

      {/* Sign In */}
      <div className="form-container sign-in">
        <SignIn />
      </div>

      {/* Toggle Container */}
      <div className="toggle-container">
        <div className="toggle">
          <div className="toggle-panel toggle-left">
            <h1>Welcome Back!</h1>
            <p>Enter Your Personal Details and start journey with us</p>
            <button className="hidden" onClick={() => setActive(false)}>Sign In</button>
          </div>
          <div className="toggle-panel toggle-right">
            <h1>Hello!</h1>
            <p>Register With Your Personal Details and start journey with us</p>
            <button className="hidden" onClick={() => setActive(true)}>Sign Up</button>
          </div>
        </div>
      </div>
    </div>
  );
}

// function Dashboard() {
//   const navigate = useNavigate();

//   const handleLogout = () => {
//     localStorage.removeItem("token"); // clear token
//     navigate("/", { replace: true }); // back to login
//   };

//   return (
//     <div style={{ textAlign: "center", marginTop: "50px" }}>
//       <h1>Welcome to your Dashboard 🎉</h1>
//       <p>This page is protected and only visible after login.</p>
//       <button
//         onClick={handleLogout}
//         style={{
//           marginTop: "20px",
//           padding: "10px 20px",
//           border: "none",
//           borderRadius: "8px",
//           background: "linear-gradient(135deg, #0d47a1, #6a1b9a)",
//           color: "#fff",
//           cursor: "pointer",
//           fontWeight: "bold",
//         }}
//       >
//         Logout
//       </button>
//     </div>
//   );
// }

// // ------------------- ProtectedRoute -------------------
// function ProtectedRoute({ children }) {
//   const token = localStorage.getItem("token");
//   if (!token) {
//     return <Navigate to="/" replace />;
//   }
//   return children;
// }