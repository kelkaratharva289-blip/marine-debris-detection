import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "../styles/globals.css";

const inter = Inter({
  subsets: ["latin"],
  variable: "--font-inter",
  display: "swap",
});

export const metadata: Metadata = {
  title: "SonicSweep — Marine Debris Detection",
  description:
    "AI-powered underwater marine debris and anomaly detection using Side-Scan Sonar imagery",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className={inter.variable}>
      <body
        className="min-h-screen font-sans"
        style={
          {
            "--app-bg":
              "radial-gradient(1200px 600px at 70% -10%, rgba(14,116,144,0.16), transparent 60%), radial-gradient(900px 600px at 15% 110%, rgba(37,99,235,0.10), transparent 55%), linear-gradient(#02080f, #02080f)",
          } as React.CSSProperties
        }
      >
        {children}
      </body>
    </html>
  );
}
