interface ScientificBackgroundProps {
  domain: "drug" | "materials" | "both";
  className?: string;
}

export function ScientificBackground({ domain, className = "" }: ScientificBackgroundProps) {
  return (
    <div className={`absolute inset-0 overflow-hidden pointer-events-none ${className}`}>
      <div className="absolute inset-0 bg-gradient-to-br from-slate-950 via-slate-900 to-slate-950" />
      
      {(domain === "drug" || domain === "both") && (
        <div className="absolute inset-0 opacity-[0.08]">
          <svg className="absolute top-0 right-0 w-[600px] h-[600px] -translate-y-1/4 translate-x-1/4" viewBox="0 0 400 400" fill="none">
            <g stroke="hsl(180 70% 50%)" strokeWidth="0.5" opacity="0.6">
              <circle cx="200" cy="200" r="40" />
              <circle cx="280" cy="160" r="25" />
              <circle cx="140" cy="240" r="30" />
              <circle cx="260" cy="280" r="20" />
              <circle cx="120" cy="140" r="22" />
              <line x1="200" y1="160" x2="280" y2="160" />
              <line x1="200" y1="240" x2="140" y2="240" />
              <line x1="240" y1="200" x2="260" y2="260" />
              <line x1="160" y1="200" x2="142" y2="162" />
            </g>
          </svg>
          
          <svg className="absolute bottom-0 left-0 w-[500px] h-[400px] translate-y-1/4 -translate-x-1/4" viewBox="0 0 300 200" fill="none">
            <path
              d="M50 100 Q80 50 120 100 Q160 150 200 100 Q240 50 280 100"
              stroke="hsl(170 60% 45%)"
              strokeWidth="1"
              fill="none"
              opacity="0.5"
            />
            <path
              d="M30 120 Q70 70 110 120 Q150 170 190 120 Q230 70 270 120"
              stroke="hsl(180 70% 50%)"
              strokeWidth="0.8"
              fill="none"
              opacity="0.4"
            />
          </svg>
        </div>
      )}
      
      {(domain === "materials" || domain === "both") && (
        <div className={`absolute inset-0 ${domain === "both" ? "opacity-[0.06]" : "opacity-[0.08]"}`}>
          <svg className="absolute top-1/4 right-0 w-[500px] h-[500px] translate-x-1/4" viewBox="0 0 400 400" fill="none">
            <g stroke="hsl(210 40% 55%)" strokeWidth="0.5" opacity="0.7">
              {[0, 1, 2, 3, 4].map((row) =>
                [0, 1, 2, 3, 4].map((col) => (
                  <g key={`${row}-${col}`}>
                    <polygon
                      points={`${80 + col * 50},${60 + row * 45} ${105 + col * 50},${45 + row * 45} ${130 + col * 50},${60 + row * 45} ${130 + col * 50},${90 + row * 45} ${105 + col * 50},${105 + row * 45} ${80 + col * 50},${90 + row * 45}`}
                    />
                  </g>
                ))
              )}
            </g>
          </svg>
          
          <svg className="absolute bottom-0 left-1/4 w-[400px] h-[300px] translate-y-1/4" viewBox="0 0 300 200" fill="none">
            <g stroke="hsl(35 70% 55%)" strokeWidth="0.8" opacity="0.5">
              <line x1="20" y1="180" x2="280" y2="140" />
              <line x1="20" y1="160" x2="280" y2="120" />
              <line x1="20" y1="140" x2="280" y2="100" />
              <line x1="20" y1="120" x2="280" y2="80" />
            </g>
            <g fill="hsl(210 50% 60%)" opacity="0.4">
              <circle cx="50" cy="170" r="3" />
              <circle cx="100" cy="155" r="3" />
              <circle cx="150" cy="140" r="3" />
              <circle cx="200" cy="125" r="3" />
              <circle cx="250" cy="110" r="3" />
            </g>
          </svg>
        </div>
      )}
      
      <div className="absolute inset-0 bg-gradient-to-t from-slate-950/80 via-transparent to-slate-950/40" />
    </div>
  );
}

export function DrugDiscoveryBackground({ className = "" }: { className?: string }) {
  return <ScientificBackground domain="drug" className={className} />;
}

export function MaterialsSciencesBackground({ className = "" }: { className?: string }) {
  return <ScientificBackground domain="materials" className={className} />;
}

export function DualDomainBackground({ className = "" }: { className?: string }) {
  return <ScientificBackground domain="both" className={className} />;
}
