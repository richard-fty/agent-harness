import { BrowserRouter, Routes, Route, Navigate } from "react-router-dom";
import { AuthPage } from "./components/auth/AuthPage";
import { RequireAuth } from "./components/auth/RequireAuth";
import { LandingPage } from "./pages/LandingPage";
import { ChatEntryPage } from "./pages/OnboardingPage";
import { PrivacyPage } from "./pages/PrivacyPage";
import { SessionPage } from "./pages/SessionPage";
import { TermsPage } from "./pages/TermsPage";

export function App() {
  return (
    <BrowserRouter
      future={{
        v7_startTransition: true,
        v7_relativeSplatPath: true,
      }}
    >
      <Routes>
        <Route path="/" element={<LandingPage />} />
        <Route path="/login" element={<AuthPage mode="login" />} />
        <Route path="/register" element={<AuthPage mode="register" />} />
        <Route path="/privacy" element={<PrivacyPage />} />
        <Route path="/terms" element={<TermsPage />} />
        <Route path="/dashboard" element={<Navigate to="/chat" replace />} />
        <Route
          path="/chat"
          element={
            <RequireAuth>
              <ChatEntryPage />
            </RequireAuth>
          }
        />
        <Route
          path="/session/:sessionId"
          element={
            <RequireAuth>
              <SessionPage />
            </RequireAuth>
          }
        />
      </Routes>
    </BrowserRouter>
  );
}
