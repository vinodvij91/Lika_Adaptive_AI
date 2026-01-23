import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import type { User } from "@shared/models/auth";

const AUTH_QUERY_KEY = ["/api/ext-auth/me"];

async function fetchUser(): Promise<User | null> {
  // DEV MODE: Return mock user for exploration
  return {
    id: "dev-user",
    email: "dev@lika.sciences",
    firstName: "Developer",
    lastName: "User",
    profileImageUrl: null,
    createdAt: new Date(),
    updatedAt: new Date(),
  };
}

async function performLogout(): Promise<void> {
  localStorage.removeItem("lika-sciences-domain");
  const response = await fetch("/api/ext-auth/logout", {
    method: "POST",
    credentials: "include",
  });
  
  if (!response.ok) {
    throw new Error("Logout failed");
  }
}

export function useAuth() {
  const queryClient = useQueryClient();
  const { data: user, isLoading } = useQuery<User | null>({
    queryKey: AUTH_QUERY_KEY,
    queryFn: fetchUser,
    retry: false,
    staleTime: 1000 * 60 * 5, // 5 minutes
  });

  const logoutMutation = useMutation({
    mutationFn: performLogout,
    onSuccess: () => {
      queryClient.setQueryData(AUTH_QUERY_KEY, null);
      queryClient.invalidateQueries({ queryKey: AUTH_QUERY_KEY });
      window.location.href = "/";
    },
  });

  return {
    user,
    isLoading,
    isAuthenticated: !!user,
    logout: logoutMutation.mutate,
    isLoggingOut: logoutMutation.isPending,
  };
}

export { AUTH_QUERY_KEY };
