import { createRoot } from "react-dom/client";
import App from "./App";
import "./index.css";

console.log("renderer loaded");
window.boot?.init();
createRoot(document.getElementById("root")!).render(<App />);
