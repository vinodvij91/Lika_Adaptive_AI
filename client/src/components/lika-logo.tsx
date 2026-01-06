export function LikaLogo({ size = "default" }: { size?: "sm" | "default" | "lg" }) {
  const dimensions = {
    sm: { icon: 28, text: "text-sm" },
    default: { icon: 36, text: "text-lg" },
    lg: { icon: 48, text: "text-2xl" },
  };
  
  const { icon, text } = dimensions[size];

  return (
    <div className="flex items-center gap-3">
      <svg
        width={icon}
        height={icon}
        viewBox="0 0 48 48"
        fill="none"
        xmlns="http://www.w3.org/2000/svg"
        className="flex-shrink-0"
      >
        <defs>
          <linearGradient id="lika-gradient" x1="0%" y1="0%" x2="100%" y2="100%">
            <stop offset="0%" stopColor="#22d3ee" />
            <stop offset="50%" stopColor="#14b8a6" />
            <stop offset="100%" stopColor="#0d9488" />
          </linearGradient>
        </defs>
        <path
          d="M24 4C24 4 8 16 8 28C8 36.837 15.163 44 24 44C32.837 44 40 36.837 40 28C40 16 24 4 24 4Z"
          fill="url(#lika-gradient)"
        />
        <path
          d="M24 12C24 12 14 20 14 28C14 33.523 18.477 38 24 38C29.523 38 34 33.523 34 28C34 20 24 12 24 12Z"
          fill="white"
          fillOpacity="0.25"
        />
        <circle cx="20" cy="26" r="3" fill="white" fillOpacity="0.6" />
        <circle cx="28" cy="30" r="2" fill="white" fillOpacity="0.4" />
      </svg>
      <span className={`${text} font-light tracking-[0.25em] uppercase text-foreground`}>
        Lika Sciences
      </span>
    </div>
  );
}

export function LikaLogoIcon({ size = 36 }: { size?: number }) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 48 48"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
      className="flex-shrink-0"
    >
      <defs>
        <linearGradient id="lika-icon-gradient" x1="0%" y1="0%" x2="100%" y2="100%">
          <stop offset="0%" stopColor="#22d3ee" />
          <stop offset="50%" stopColor="#14b8a6" />
          <stop offset="100%" stopColor="#0d9488" />
        </linearGradient>
      </defs>
      <path
        d="M24 4C24 4 8 16 8 28C8 36.837 15.163 44 24 44C32.837 44 40 36.837 40 28C40 16 24 4 24 4Z"
        fill="url(#lika-icon-gradient)"
      />
      <path
        d="M24 12C24 12 14 20 14 28C14 33.523 18.477 38 24 38C29.523 38 34 33.523 34 28C34 20 24 12 24 12Z"
        fill="white"
        fillOpacity="0.25"
      />
      <circle cx="20" cy="26" r="3" fill="white" fillOpacity="0.6" />
      <circle cx="28" cy="30" r="2" fill="white" fillOpacity="0.4" />
    </svg>
  );
}
