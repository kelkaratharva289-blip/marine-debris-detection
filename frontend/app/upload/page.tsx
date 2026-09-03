import PageShell from "@/components/layout/PageShell";
import UploadForm from "@/components/detection/UploadForm";

export default function Upload() {
  return (
    <PageShell className="flex justify-center">
      <UploadForm />
    </PageShell>
  );
}
