"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

const NAV_LINKS = [
  { href: "/dashboard", label: "Dashboard" },
  { href: "/scans", label: "Scans" },
  { href: "/analyze", label: "Analyze" },
  { href: "/upload", label: "Upload" },
];

export default function Navbar() {
  const pathname = usePathname();

  return (
    <header className="sticky top-0 z-50 border-b border-white/[0.06] bg-abyss-950/70 backdrop-blur-md">
      <div className="mx-auto flex h-16 max-w-7xl items-center justify-between px-6">
        <Link href="/" className="group flex items-center gap-3">
          <div className="flex h-9 w-9 items-center justify-center rounded-md border border-cyan-400/25 bg-cyan-400/10 text-sm font-extrabold text-cyan-300 transition-colors duration-150 group-hover:bg-cyan-400/20">
            SS
          </div>
          <div className="leading-tight">
            <span className="block text-lg font-bold tracking-tight text-slate-50">
              SonicSweep
            </span>
            <span className="block text-[10px] font-medium uppercase tracking-[0.2em] text-slate-500">
              Marine Intelligence
            </span>
          </div>
        </Link>

        <nav className="flex items-center gap-1">
          {NAV_LINKS.map((link) => {
            const active =
              pathname === link.href ||
              (link.href !== "/dashboard" && pathname.startsWith(link.href));
            return (
              <Link
                key={link.href}
                href={link.href}
                className={`relative rounded-md px-4 py-2 text-sm font-medium transition-colors duration-150 ${
                  active
                    ? "text-cyan-300"
                    : "text-slate-400 hover:text-slate-200"
                }`}
              >
                {link.label}
                {active && (
                  <span className="absolute inset-x-4 -bottom-[13px] h-px bg-cyan-400/60" />
                )}
              </Link>
            );
          })}
        </nav>
      </div>
    </header>
  );
}
