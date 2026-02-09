import { useEffect } from "react";
import { useLocation } from "wouter";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { LikaLogoLeafGradient } from "@/components/lika-logo";
import { Loader2, LogIn } from "lucide-react";
import { useAuth } from "@/hooks/use-auth";

export default function LoginPage() {
  const [, setLocation] = useLocation();
  const { user, isLoading } = useAuth();

  useEffect(() => {
    if (user) {
      setLocation("/dashboard");
    }
  }, [user, setLocation]);

  if (isLoading) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-slate-950 via-slate-900 to-slate-950 flex items-center justify-center">
        <Loader2 className="h-8 w-8 animate-spin text-cyan-400" />
      </div>
    );
  }

  function handleLogin() {
    window.location.href = "/api/auth/login";
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-950 via-slate-900 to-slate-950 flex items-center justify-center p-6">
      <div className="absolute inset-0 overflow-hidden pointer-events-none">
        <div className="absolute top-1/4 left-1/4 w-96 h-96 bg-cyan-500/10 rounded-full blur-3xl" />
        <div className="absolute bottom-1/4 right-1/4 w-80 h-80 bg-amber-500/10 rounded-full blur-3xl" />
      </div>

      <Card className="w-full max-w-md relative bg-slate-900/80 backdrop-blur-md border-slate-800">
        <CardHeader className="text-center space-y-4">
          <div className="flex justify-center">
            <LikaLogoLeafGradient size={56} />
          </div>
          <div>
            <CardTitle className="text-2xl font-light tracking-[0.15em] uppercase text-white">
              Sign In
            </CardTitle>
            <CardDescription className="text-slate-400 mt-2">
              Access your Lika Sciences workspace
            </CardDescription>
          </div>
        </CardHeader>
        <CardContent className="space-y-6">
          <Button
            onClick={handleLogin}
            className="w-full bg-gradient-to-r from-cyan-600 to-teal-600 hover:from-cyan-500 hover:to-teal-500 border-0 shadow-lg shadow-cyan-500/20"
            data-testid="button-login-auth0"
          >
            <LogIn className="mr-2 h-4 w-4" />
            Sign in with Auth0
          </Button>

          <div className="mt-6 text-center">
            <a
              href="/"
              className="text-sm text-slate-400 hover:text-cyan-400 transition-colors"
              data-testid="link-back-home"
            >
              Back to home
            </a>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
