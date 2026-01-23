import { createContext, useContext, useState, useCallback, useEffect, type ReactNode } from "react";

export type DiscoveryDomain = "drug" | "materials";

interface DomainContextValue {
  domain: DiscoveryDomain | null;
  setDomain: (domain: DiscoveryDomain) => void;
  clearDomain: () => void;
  isDrugDomain: boolean;
  isMaterialsDomain: boolean;
  hasDomainSelected: boolean;
}

const DomainContext = createContext<DomainContextValue | undefined>(undefined);

const DOMAIN_STORAGE_KEY = "lika-sciences-domain";

export function DomainProvider({ children }: { children: ReactNode }) {
  const [domain, setDomainState] = useState<DiscoveryDomain | null>(() => {
    if (typeof window !== "undefined") {
      const stored = localStorage.getItem(DOMAIN_STORAGE_KEY);
      if (stored === "drug" || stored === "materials") {
        return stored;
      }
    }
    return null;
  });

  useEffect(() => {
    if (domain) {
      document.documentElement.setAttribute("data-domain", domain);
    } else {
      document.documentElement.removeAttribute("data-domain");
    }
  }, [domain]);

  const setDomain = useCallback((newDomain: DiscoveryDomain) => {
    setDomainState(newDomain);
    localStorage.setItem(DOMAIN_STORAGE_KEY, newDomain);
  }, []);

  const clearDomain = useCallback(() => {
    setDomainState(null);
    localStorage.removeItem(DOMAIN_STORAGE_KEY);
  }, []);

  const value: DomainContextValue = {
    domain,
    setDomain,
    clearDomain,
    isDrugDomain: domain === "drug",
    isMaterialsDomain: domain === "materials",
    hasDomainSelected: domain !== null,
  };

  return (
    <DomainContext.Provider value={value}>
      {children}
    </DomainContext.Provider>
  );
}

export function useDomain() {
  const context = useContext(DomainContext);
  if (!context) {
    throw new Error("useDomain must be used within a DomainProvider");
  }
  return context;
}
