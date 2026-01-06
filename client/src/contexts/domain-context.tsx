import { createContext, useContext, useState, useCallback, type ReactNode } from "react";

export type DiscoveryDomain = "drug" | "materials";

interface DomainContextValue {
  domain: DiscoveryDomain;
  setDomain: (domain: DiscoveryDomain) => void;
  isDrugDomain: boolean;
  isMaterialsDomain: boolean;
}

const DomainContext = createContext<DomainContextValue | undefined>(undefined);

export function DomainProvider({ children, defaultDomain = "drug" }: { children: ReactNode; defaultDomain?: DiscoveryDomain }) {
  const [domain, setDomainState] = useState<DiscoveryDomain>(defaultDomain);

  const setDomain = useCallback((newDomain: DiscoveryDomain) => {
    setDomainState(newDomain);
    document.documentElement.setAttribute("data-domain", newDomain);
  }, []);

  const value: DomainContextValue = {
    domain,
    setDomain,
    isDrugDomain: domain === "drug",
    isMaterialsDomain: domain === "materials",
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
