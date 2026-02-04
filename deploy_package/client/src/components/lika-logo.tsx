interface LogoProps {
  size?: number;
  className?: string;
}

export function LikaLogoLeafMono({ size = 36, className = "" }: LogoProps) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 48 48"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
      className={className}
    >
      <path
        d="M18 6C18 6 6 14 6 26C6 34 12 42 20 44C20 44 16 36 16 28C16 18 24 10 24 10C24 10 18 6 18 6Z"
        fill="currentColor"
      />
      <path
        d="M30 6C30 6 42 14 42 26C42 34 36 42 28 44C28 44 32 36 32 28C32 18 24 10 24 10C24 10 30 6 30 6Z"
        fill="currentColor"
      />
    </svg>
  );
}

export function LikaLogoLeafGradient({ size = 36, className = "" }: LogoProps) {
  const gradientId = `lika-leaf-gradient-${Math.random().toString(36).substr(2, 9)}`;
  const gradientId2 = `lika-leaf-gradient2-${Math.random().toString(36).substr(2, 9)}`;
  
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 48 48"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
      className={className}
    >
      <defs>
        <linearGradient id={gradientId} x1="6" y1="6" x2="24" y2="44" gradientUnits="userSpaceOnUse">
          <stop offset="0%" stopColor="#22d3ee" />
          <stop offset="50%" stopColor="#14b8a6" />
          <stop offset="100%" stopColor="#10b981" />
        </linearGradient>
        <linearGradient id={gradientId2} x1="24" y1="6" x2="42" y2="44" gradientUnits="userSpaceOnUse">
          <stop offset="0%" stopColor="#f59e0b" />
          <stop offset="50%" stopColor="#eab308" />
          <stop offset="100%" stopColor="#84cc16" />
        </linearGradient>
      </defs>
      <path
        d="M18 6C18 6 6 14 6 26C6 34 12 42 20 44C20 44 16 36 16 28C16 18 24 10 24 10C24 10 18 6 18 6Z"
        fill={`url(#${gradientId})`}
      />
      <path
        d="M30 6C30 6 42 14 42 26C42 34 36 42 28 44C28 44 32 36 32 28C32 18 24 10 24 10C24 10 30 6 30 6Z"
        fill={`url(#${gradientId2})`}
      />
    </svg>
  );
}

export function LikaLogoLeafCompact({ size = 24, className = "" }: LogoProps) {
  const gradientId = `lika-compact-gradient-${Math.random().toString(36).substr(2, 9)}`;
  
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 48 48"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
      className={className}
    >
      <defs>
        <linearGradient id={gradientId} x1="6" y1="6" x2="42" y2="44" gradientUnits="userSpaceOnUse">
          <stop offset="0%" stopColor="#22d3ee" />
          <stop offset="100%" stopColor="#f59e0b" />
        </linearGradient>
      </defs>
      <path
        d="M18 6C18 6 6 14 6 26C6 34 12 42 20 44C20 44 16 36 16 28C16 18 24 10 24 10C24 10 18 6 18 6Z"
        fill={`url(#${gradientId})`}
      />
      <path
        d="M30 6C30 6 42 14 42 26C42 34 36 42 28 44C28 44 32 36 32 28C32 18 24 10 24 10C24 10 30 6 30 6Z"
        fill={`url(#${gradientId})`}
      />
    </svg>
  );
}

interface LikaLogoFullProps {
  size?: "sm" | "default" | "lg";
  variant?: "mono" | "gradient";
  className?: string;
}

export function LikaLogo({ size = "default", variant = "gradient", className = "" }: LikaLogoFullProps) {
  const dimensions = {
    sm: { icon: 24, text: "text-xs", spacing: "tracking-[0.2em]" },
    default: { icon: 32, text: "text-sm", spacing: "tracking-[0.25em]" },
    lg: { icon: 40, text: "text-base", spacing: "tracking-[0.25em]" },
  };
  
  const { icon, text, spacing } = dimensions[size];
  const LogoIcon = variant === "gradient" ? LikaLogoLeafGradient : LikaLogoLeafMono;

  return (
    <div className={`flex items-center gap-2.5 ${className}`}>
      <LogoIcon size={icon} />
      <span className={`${text} font-medium ${spacing} uppercase text-foreground`}>
        Lika Sciences
      </span>
    </div>
  );
}

export function LikaLogoIcon({ size = 36 }: { size?: number }) {
  return <LikaLogoLeafGradient size={size} />;
}
