"use client";

import { motion } from "framer-motion";
import { Database, Ruler, Calendar, FileText } from "lucide-react";
import { DataSummary } from "@/types";

interface SummaryCardsProps {
  summary?: DataSummary;
  intent?: string;
}

export function SummaryCards({ summary, intent }: SummaryCardsProps) {
  if (!summary) return null;

  const cards = [
    {
      icon: Database,
      label: "Profiles",
      value: summary.matched_records ?? 0,
      color: "text-ocean-400",
      bg: "bg-ocean-500/10",
      border: "border-ocean-500/20",
    },
    {
      icon: Ruler,
      label: "Measurements",
      value: summary.total_measurements ?? 0,
      color: "text-emerald-400",
      bg: "bg-emerald-500/10",
      border: "border-emerald-500/20",
    },
    {
      icon: Calendar,
      label: "Date Range",
      value:
        summary.date_range?.min && summary.date_range?.max
          ? `${summary.date_range.min.slice(0, 10)} → ${summary.date_range.max.slice(0, 10)}`
          : "N/A",
      color: "text-amber-400",
      bg: "bg-amber-500/10",
      border: "border-amber-500/20",
      isText: true,
    },
    {
      icon: FileText,
      label: "Intent",
      value: intent ? intent.replace("_", " ") : "—",
      color: "text-violet-400",
      bg: "bg-violet-500/10",
      border: "border-violet-500/20",
      isText: true,
    },
  ];

  return (
    <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
      {cards.map((card, i) => (
        <motion.div
          key={card.label}
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.3, delay: i * 0.08 }}
          className={`flex items-center gap-3 px-4 py-3 rounded-lg border ${card.bg} ${card.border}`}
        >
          <card.icon className={`w-4 h-4 ${card.color}`} />
          <div className="min-w-0">
            <p className="text-[10px] uppercase tracking-wider text-surface-500 font-semibold">
              {card.label}
            </p>
            <p className={`text-sm font-semibold text-surface-100 truncate ${card.isText ? "text-xs" : ""}`}>
              {typeof card.value === "number" ? card.value.toLocaleString() : card.value}
            </p>
          </div>
        </motion.div>
      ))}
    </div>
  );
}
