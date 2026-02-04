import { useState, useEffect } from "react";
import { Button } from "@/components/ui/button";
import { ThemeToggle } from "@/components/theme-toggle";
import { LikaLogoLeafGradient } from "@/components/lika-logo";
import { useAuth } from "@/hooks/use-auth";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { Avatar, AvatarFallback } from "@/components/ui/avatar";
import { ChevronDown, LogOut, User, Building2 } from "lucide-react";
import { Link } from "wouter";

export function Navbar() {
  const [scrolled, setScrolled] = useState(false);
  const { user, isAuthenticated, logout, isLoggingOut } = useAuth();

  useEffect(() => {
    const handleScroll = () => {
      setScrolled(window.scrollY > 20);
    };
    window.addEventListener("scroll", handleScroll);
    return () => window.removeEventListener("scroll", handleScroll);
  }, []);

  return (
    <header
      className={`fixed top-0 left-0 right-0 z-50 transition-all duration-300 ${
        scrolled
          ? "bg-[rgba(8,18,40,0.95)] backdrop-blur-md border-b border-cyan-500/20"
          : "bg-[rgba(8,18,40,0.85)] backdrop-blur-sm border-b border-cyan-500/10"
      }`}
    >
      <div className="max-w-7xl mx-auto px-6 h-16 flex items-center justify-between gap-4">
        <Link href="/" className="flex items-center gap-2.5 group">
          <LikaLogoLeafGradient size={32} className="transition-transform group-hover:scale-105" />
          <span className="text-sm font-medium tracking-[0.25em] uppercase text-white/90 group-hover:text-white transition-colors">
            Lika Sciences
          </span>
        </Link>

        <nav className="hidden md:flex items-center gap-8">
          <NavLink href="#features">Features</NavLink>
          <NavLink href="#drug-discovery">Drug Discovery</NavLink>
          <NavLink href="#materials">Materials Science</NavLink>
          <NavLink href="/use-cases">Use Cases</NavLink>
        </nav>

        <div className="flex items-center gap-3">
          <ThemeToggle />
          
          {isAuthenticated && user ? (
            <DropdownMenu>
              <DropdownMenuTrigger asChild>
                <Button variant="ghost" className="gap-2 text-white/80 hover:text-white hover:bg-white/10" data-testid="button-user-menu">
                  <Avatar className="h-7 w-7">
                    <AvatarFallback className="bg-gradient-to-br from-cyan-500 to-teal-600 text-white text-xs">
                      {(user.firstName || user.email || "U").slice(0, 2).toUpperCase()}
                    </AvatarFallback>
                  </Avatar>
                  <span className="hidden sm:inline text-sm">{user.firstName || user.email || "User"}</span>
                  <ChevronDown className="h-4 w-4" />
                </Button>
              </DropdownMenuTrigger>
              <DropdownMenuContent align="end" className="w-56">
                <div className="px-2 py-1.5">
                  <p className="text-sm font-medium">{user.firstName || user.email || "User"}</p>
                </div>
                <DropdownMenuSeparator />
                <DropdownMenuItem asChild>
                  <Link href="/dashboard" className="flex items-center gap-2 w-full cursor-pointer">
                    <User className="h-4 w-4" />
                    Dashboard
                  </Link>
                </DropdownMenuItem>
                <DropdownMenuSeparator />
                <DropdownMenuItem onClick={() => logout()} disabled={isLoggingOut} className="text-destructive cursor-pointer" data-testid="button-logout">
                  <LogOut className="h-4 w-4 mr-2" />
                  Sign Out
                </DropdownMenuItem>
              </DropdownMenuContent>
            </DropdownMenu>
          ) : (
            <Link href="/login" asChild>
              <Button 
                className="bg-gradient-to-r from-cyan-600 to-teal-600 hover:from-cyan-500 hover:to-teal-500 border-0 shadow-lg shadow-cyan-500/20 hover:shadow-cyan-500/30 transition-all"
                data-testid="button-login"
              >
                Sign In
              </Button>
            </Link>
          )}
        </div>
      </div>
    </header>
  );
}

function NavLink({ href, children }: { href: string; children: React.ReactNode }) {
  return (
    <a
      href={href}
      className="text-sm text-white/60 hover:text-cyan-400 transition-colors relative group"
    >
      {children}
      <span className="absolute -bottom-1 left-0 w-0 h-0.5 bg-gradient-to-r from-cyan-400 to-teal-400 group-hover:w-full transition-all duration-300" />
    </a>
  );
}
