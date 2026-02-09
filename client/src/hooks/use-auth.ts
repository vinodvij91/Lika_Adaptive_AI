import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import type { User } from "@shared/models/auth";

export interface AuthUser extends Partial<User> {
  id: string;
  userId?: string;
  email: string | null;
  firstName?: string | null;
  lastName?: string | null;
  profileImageUrl?: string | null;
  tenantId: string;
  role: string;
  authenticated: boolean;
}

const AUTH_QUERY_KEY = ["/api/auth/me"];

async function fetchUser(): Promise<AuthUser | null> {
  try {
    const response = await fetch("/api/auth/me", {
      credentials: "include",
    });

    if (!response.ok) {
      return null;
    }

    return await response.json();
  } catch {
    return null;
  }
}

async function performLogout(): Promise<void> {
  localStorage.removeItem("lika-sciences-domain");
  window.location.href = "/api/auth/logout";
}

export function useAuth() {
  const queryClient = useQueryClient();
  const { data: user, isLoading } = useQuery<AuthUser | null>({
    queryKey: AUTH_QUERY_KEY,
    queryFn: fetchUser,
    retry: false,
    staleTime: 1000 * 60 * 5,
  });

  const logoutMutation = useMutation({
    mutationFn: performLogout,
    onSuccess: () => {
      queryClient.setQueryData(AUTH_QUERY_KEY, null);
      queryClient.invalidateQueries({ queryKey: AUTH_QUERY_KEY });
    },
  });

  return {
    user,
    isLoading,
    isAuthenticated: !!user?.authenticated,
    tenantId: user?.tenantId || null,
    role: user?.role || null,
    logout: logoutMutation.mutate,
    isLoggingOut: logoutMutation.isPending,
  };
}

export { AUTH_QUERY_KEY };
