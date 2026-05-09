import SectionMasthead from "@/components/editorial/SectionMasthead"
import StaggerGroup from "@/components/ui/StaggerGroup"

interface SkillsProps {
  skills?: string[]
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
    <section className="zone-teal py-16 md:py-24">
      {showMasthead && (
        <SectionMasthead
          kicker="The Toolbox"
          title="Skills and tools I work with"
          description="The intersection of statistical thinking, signal processing, and software — picked up across school, work, and side projects."
        />
      )}

      <StaggerGroup
        staggerDelay={60}
        direction="scale"
        distance={0}
        className="flex flex-wrap gap-x-3 gap-y-3"
      >
        {skills.map((skill, idx) => (
          <span
            key={skill}
            className="glass-card inline-flex items-center gap-2 rounded-lg px-3.5 py-1.5 text-sm text-foreground glow-hover"
          >
            <span className="font-mono text-[10px] text-muted-foreground">
              {String(idx + 1).padStart(2, "0")}
            </span>
            {skill}
          </span>
        ))}
      </StaggerGroup>
    </section>
  )
}
