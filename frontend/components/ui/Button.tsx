interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: "primary" | "ghost";
  size?: "sm" | "md";
  loading?: boolean;
}

export default function Button({
  variant = "primary",
  size = "md",
  loading = false,
  className = "",
  children,
  disabled,
  ...rest
}: ButtonProps) {
  const base =
    "inline-flex items-center justify-center gap-2 rounded-md font-semibold transition-colors duration-150 focus:outline-none focus:ring-2 disabled:cursor-not-allowed disabled:opacity-50";
  const sizeClass = size === "sm" ? "px-3 py-1.5 text-xs" : "px-4 py-2 text-sm";
  const variantClass =
    variant === "primary"
      ? "bg-cyan-500 text-abyss-950 hover:bg-cyan-400 focus:ring-cyan-400/40"
      : "border border-white/10 bg-white/[0.03] text-slate-200 hover:border-cyan-400/40 hover:text-white focus:ring-cyan-400/40";

  return (
    <button
      className={`${base} ${sizeClass} ${variantClass} ${className}`}
      disabled={disabled || loading}
      {...rest}
    >
      {loading && (
        <span className="inline-block h-3.5 w-3.5 animate-spin rounded-full border-2 border-white/30 border-b-white border-t-white" />
      )}
      {children}
    </button>
  );
}
