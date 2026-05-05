import SectionMasthead from "@/components/editorial/SectionMasthead"

interface SkillsProps {
  skills?: string[]
  /**
   * When `false`, render only the chip cloud without the section masthead.
   * Used on the About page where its own H1 already frames the section.
   */
  showMasthead?: boolean
}

export default function Skills({
  skills = [
    "Python",
    "Machine Learning",
    "Deep Learning",
    "Computer Vision",
    "Signal Processing",
    "Image Processing",
    "Data Preprocessing",
    "Data Analysis",
    "Data Visualization",
  ],
  showMasthead = true,
}: SkillsProps) {
  return (
    <section className="py-16 md:py-24">
      {showMasthead && (
        <SectionMasthead
          kicker="The Toolbox"
          title="Skills and tools I work with"
          description="The intersection of statistical thinking, signal processing, and software — picked up across school, work, and side projects."
        />
      )}

      <div className="flex flex-wrap gap-x-3 gap-y-3">
        {skills.map((skill, idx) => (
          <span
            key={skill}
            className="inline-flex items-center gap-2 border border-border bg-card px-3.5 py-1.5 text-sm text-foreground transition-colors hover:border-accent hover:text-accent"
          >
            <span className="font-mono text-[10px] text-muted-foreground">
              {String(idx + 1).padStart(2, "0")}
            </span>
            {skill}
          </span>
        ))}
      </div>
    </section>
  )
}
