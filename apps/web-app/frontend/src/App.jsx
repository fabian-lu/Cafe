import { Routes, Route, Navigate } from "react-router-dom";
import Sidebar from "./components/Sidebar.jsx";
import Techniques from "./pages/Techniques.jsx";
import Studies from "./pages/Studies.jsx";
import StudyDetail from "./pages/StudyDetail.jsx";
import Questions from "./pages/Questions.jsx";
import NewDataset from "./pages/NewDataset.jsx";
import ViewDataset from "./pages/ViewDataset.jsx";
import Rubrics from "./pages/Rubrics.jsx";
import ViewRubric from "./pages/ViewRubric.jsx";
import Results from "./pages/Results.jsx";
import Raters from "./pages/Raters.jsx";

export default function App() {
  return (
    <div className="shell">
      <Sidebar />
      <main className="content">
        <div className="content-inner">
          <Routes>
            <Route path="/" element={<Navigate to="/studies" replace />} />
            <Route path="/studies" element={<Studies />} />
            <Route path="/studies/:id" element={<StudyDetail />} />
            <Route path="/questions" element={<Questions />} />
            <Route path="/questions/new" element={<NewDataset />} />
            <Route path="/questions/:id" element={<ViewDataset />} />
            <Route path="/rubrics" element={<Rubrics />} />
            <Route path="/rubrics/:id" element={<ViewRubric />} />
            <Route path="/techniques" element={<Techniques />} />
            <Route path="/results" element={<Results />} />
            <Route path="/raters" element={<Raters />} />
          </Routes>
        </div>
      </main>
    </div>
  );
}
