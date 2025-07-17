// components/CTA.tsx
import Link from "next/link"
import { FileDown } from "lucide-react"
import { Button } from "@/components/ui/button"

export default function CTA() {
  return (
    <section className="py-16">
      <div className="bg-muted rounded-lg p-8 md:p-12 text-center">
        <h2 className="text-3xl font-bold mb-4">Open to Opportunities</h2>
        <p className="text-muted-foreground mb-8 max-w-2xl mx-auto">
        I build end-to-end intelligent systems. Seeking a full-time role where my skills in AI and Data Engineering can create immediate impact. Open to new challenges.
        </p>
        <div className="flex flex-col sm:flex-row justify-center gap-4">
          {/* "About Me" Button (already modified correctly) */}
          <Button asChild size="lg">
            <Link href="/about">About Me</Link>
          </Button>

          {/* "Download Resume" Button - Apply the fix here */}
          <Button asChild variant="outline" size="lg">
            <a href="/resume.pdf" target="_blank">
              {/* Wrap icon and text in a single span */}
              <span className="flex items-center justify-center"> {/* <-- ADD THIS WRAPPER */}
                <FileDown className="mr-2 h-4 w-4" />
                Download Resume
              </span> {/* <-- CLOSE THE WRAPPER */}
            </a>
          </Button>
        </div>
      </div>
    </section>
  )
}