"use client";

import { motion } from "framer-motion";
import { Anchor, MapPin, Calendar, Activity, Hash } from "lucide-react";
import { MapData } from "@/types";

interface FloatDetailCardProps {
  float: MapData;
  onClear: () => void;
}

export function FloatDetailCard({ float, onClear }: FloatDetailCardProps) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: 8 }}
      className="rounded-lg border border-ocean-500/20 bg-ocean-500/5 p-4"
    >
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2">
          <Anchor className="w-4 h-4 text-ocean-400" />
          <h3 className="text-sm font-semibold text-surface-100">
            Selected Float
          </h3>
        </div>
        <button
          onClick={onClear}
          className="text-xs text-surface-500 hover:text-surface-300 transition-colors px-2 py-1 rounded hover:bg-surface-800"
        >
          View All
        </button>
      </div>

      <div className="grid grid-cols-2 lg:grid-cols-3 gap-3">
        <div className="flex items-center gap-2">
          <Hash className="w-3.5 h-3.5 text-surface-500" />
          <div>
            <p className="text-[10px] uppercase tracking-wider text-surface-600 font-semibold">
              Float ID
            </p>
            <p className="text-sm font-mono font-semibold text-ocean-300">
              {float.float_id}
            </p>
          </div>
        </div>

        <div className="flex items-center gap-2">
          <Activity className="w-3.5 h-3.5 text-surface-500" />
          <div>
            <p className="text-[10px] uppercase tracking-wider text-surface-600 font-semibold">
              DAC
            </p>
            <p className="text-sm font-semibold text-surface-200">
              {float.dac.toUpperCase()}
            </p>
          </div>
        </div>

        <div className="flex items-center gap-2">
          <Calendar className="w-3.5 h-3.5 text-surface-500" />
          <div>
            <p className="text-[10px] uppercase tracking-wider text-surface-600 font-semibold">
              Date
            </p>
            <p className="text-sm font-semibold text-surface-200">
              {float.profile_date
                ? new Date(float.profile_date).toLocaleDateString()
                : "N/A"}
            </p>
          </div>
        </div>

        <div className="flex items-center gap-2">
          <MapPin className="w-3.5 h-3.5 text-surface-500" />
          <div>
            <p className="text-[10px] uppercase tracking-wider text-surface-600 font-semibold">
              Latitude
            </p>
            <p className="text-sm font-mono font-semibold text-surface-200">
              {float.latitude.toFixed(3)}°
            </p>
          </div>
        </div>

        <div className="flex items-center gap-2">
          <MapPin className="w-3.5 h-3.5 text-surface-500" />
          <div>
            <p className="text-[10px] uppercase tracking-wider text-surface-600 font-semibold">
              Longitude
            </p>
            <p className="text-sm font-mono font-semibold text-surface-200">
              {float.longitude.toFixed(3)}°
            </p>
          </div>
        </div>

        <div className="flex items-center gap-2">
          <Activity className="w-3.5 h-3.5 text-surface-500" />
          <div className="min-w-0">
            <p className="text-[10px] uppercase tracking-wider text-surface-600 font-semibold">
              Variables
            </p>
            <p className="text-sm font-semibold text-surface-200 truncate">
              {float.variables.slice(0, 4).join(", ")}
              {float.variables.length > 4 && "+"}
            </p>
          </div>
        </div>
      </div>
    </motion.div>
  );
}
