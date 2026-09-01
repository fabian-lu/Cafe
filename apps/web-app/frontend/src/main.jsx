import React from "react";
import ReactDOM from "react-dom/client";
import { BrowserRouter } from "react-router-dom";
import App from "./App.jsx";
import { ConfirmProvider } from "./components/ConfirmModal.jsx";
import "./index.css";
// Optional theme OVERWRITE: with VITE_THEME=diva the DIVA token file loads on top of
// index.css; without it, nothing is imported and the default Factorial Mono stays.
if (import.meta.env.VITE_THEME === "diva") import("./themes/diva.css");

// basename honours Vite's `base` (e.g. "/demo/" for the static demo build, "/" otherwise)
const basename = import.meta.env.BASE_URL.replace(/\/$/, "") || "/";

ReactDOM.createRoot(document.getElementById("root")).render(
  <React.StrictMode>
    <BrowserRouter basename={basename}>
      <ConfirmProvider>
        <App />
      </ConfirmProvider>
    </BrowserRouter>
  </React.StrictMode>
);
