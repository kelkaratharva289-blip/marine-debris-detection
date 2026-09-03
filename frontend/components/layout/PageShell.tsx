import Navbar from "./Navbar";

interface PageShellProps {
  children: React.ReactNode;
  className?: string;
}

export default function PageShell({ children, className = "" }: PageShellProps) {
  return (
    <div className="flex min-h-screen flex-col">
      <Navbar />
      <main className={`mx-auto w-full max-w-7xl flex-1 px-6 py-8 ${className}`}>
        {children}
      </main>
    </div>
  );
}
