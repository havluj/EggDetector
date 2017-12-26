package org.cvut.havluja1.eggdetector;

interface Classifier {

    class Recognition {

        private final int id;

        private final String title;

        private final Float confidence;

        private RectF location;

        public Recognition(
                final int id, final String title, final Float confidence, final RectF location) {
            this.id = id;
            this.title = title;
            this.confidence = confidence;
            this.location = location;
        }

        public int getId() {
            return id;
        }

        public String getTitle() {
            return title;
        }

        public Float getConfidence() {
            return confidence;
        }

        public RectF getLocation() {
            return new RectF(location);
        }

        public void setLocation(RectF location) {
            this.location = location;
        }

        @Override
        public String toString() {
            return "Recognition{" +
                    "id=" + id +
                    ", title='" + title + '\'' +
                    ", confidence=" + confidence +
                    ", location=" + location +
                    '}';
        }
    }

    void close();
}