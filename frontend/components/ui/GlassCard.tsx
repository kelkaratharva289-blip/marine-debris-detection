import Link from "next/link";

interface GlassCardProps {
  children: React.ReactNode;
  className?: string;
  pad?: "none" | "sm" | "md" | "lg";
  hover?: boolean;
}

const PADDING = {
  none: "",
  sm: "p-4",
  md: "p-5",
  lg: "p-8",
};

export default function GlassCard({
  children,
  className = "",
  pad = "md",
  hover = false,
}: GlassCardProps) {
  return (
    <div
      className={`glass rounded-lg transition-colors duration-150 ${
        hover ? "hover:border-cyan-400/20" : ""
      } ${PADDING[pad]} ${className}`}
    >
      {children}
    </div>
  );
}

export function GlassLink({
  href,
  children,
  className = "",
}: {
  href: string;
  children: React.ReactNode;
  className?: string;
}) {
  return (
    <Link
      href={href}
      className={`glass rounded-lg transition-colors duration-150 hover:border-cyan-400/25 ${className}`}
    >
      {children}
    </Link>
  );
}
