export interface MapData {
  float_id: string;
  latitude: number;
  longitude: number;
  profile_date: string | null;
  dac: string;
  variables: string[];
  selected: boolean;
}

export interface ChatRequest {
  message: string;
  session_id?: string;
}

export interface ChatResponse {
  intent: string;
  message: string;
  figure: PlotlyFigure | null;
  data_summary: DataSummary;
  map_data: MapData[];
}

export interface PlotlyFigure {
  data: PlotlyTrace[];
  layout: Record<string, unknown>;
}

export interface PlotlyTrace {
  x: number[];
  y: number[];
  mode: string;
  name: string;
  type?: string;
  line?: Record<string, unknown>;
  marker?: Record<string, unknown>;
  hovertext?: string[];
  hoverinfo?: string;
  showlegend?: boolean;
  xaxis?: string;
  yaxis?: string;
}

export interface DataSummary {
  matched_records?: number;
  total_measurements?: number;
  unique_profiles?: number;
  date_range?: {
    min: string | null;
    max: string | null;
  };
  files?: string[];
  readable?: number;
}

export interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  timestamp: Date;
  figure?: PlotlyFigure | null;
  summary?: DataSummary;
  intent?: string;
  mapData?: MapData[];
  isLoading?: boolean;
  error?: string;
}

export interface HealthResponse {
  status: string;
  metadata_loaded: boolean;
}
