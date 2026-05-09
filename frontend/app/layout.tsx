import type { Metadata } from "next";
import "./globals.css";
import Nav from "@/components/Nav";
import { SlateProvider } from "@/app/context/SlateContext";

export const metadata: Metadata = {
  title: "DFS War Room",
  description: "MLB DraftKings GPP Lineup Optimizer",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <head>
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link
          rel="preconnect"
          href="https://fonts.gstatic.com"
          crossOrigin=""
        />
        <link
          href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600;700&display=swap"
          rel="stylesheet"
        />
      </head>
      <body>
        <SlateProvider>
          <div className="min-h-screen flex flex-col" style={{ backgroundColor: "#0a0e1a" }}>
            <Nav />
            <main className="flex-1 overflow-auto">{children}</main>
          </div>
        </SlateProvider>
      </body>
    </html>
  );
}
