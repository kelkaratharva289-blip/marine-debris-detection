import { Suspense } from "react";
import PageShell from "@/components/layout/PageShell";
import DetectionDetail from "@/components/detection/DetectionDetail";

export default function ScanDetail({ params }: { params: { id: string } }) {
  return (
    <PageShell>
      <Suspense
        fallback={
          <div className="mx-auto w-full max-w-4xl space-y-6">
            <div className="h-6 w-32 animate-pulse rounded-md bg-white/[0.05]" aria-hidden />
            <div className="h-9 w-72 animate-pulse rounded-md bg-white/[0.05]" aria-hidden />
            <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
              {[0, 1, 2, 3].map((i) => (
                <div key={i} className="h-24 animate-pulse rounded-md bg-white/[0.05]" aria-hidden />
              ))}
            </div>
            <div className="h-72 animate-pulse rounded-md bg-white/[0.05]" aria-hidden />
          </div>
        }
      >
        <DetectionDetail scanId={params.id} />
      </Suspense>
    </PageShell>
  );
}