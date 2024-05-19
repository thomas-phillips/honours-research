CREATE TABLE IF NOT EXISTS "model" (
    "id" bigint GENERATED ALWAYS AS IDENTITY NOT NULL UNIQUE,
    "name" text NOT NULL,
    "epochs" bigint NOT NULL,
    "inclusion" bigint NOT NULL,
    "exclusion" bigint NOT NULL,
    "input_channels" bigint NOT NULL,
    "preprocessing" text NOT NULL,
    "type" text NOT NULL,
    PRIMARY KEY ("id")
);

CREATE TABLE IF NOT EXISTS "cnn_data" (
    "id" bigint GENERATED ALWAYS AS IDENTITY NOT NULL UNIQUE,
    "learning_rate" varchar(255) NOT NULL,
    "lr_schd_gamma" varchar(255) NOT NULL,
    "optimiser" text NOT NULL,
    "early_stop" bigint NOT NULL,
    "batch_size" bigint NOT NULL,
    "shot" bigint,
    "shuffle" boolean,
    "model_id" bigint NOT NULL,
    PRIMARY KEY ("id"),
    CONSTRAINT "cnn_data_fk8" FOREIGN KEY ("model_id") REFERENCES "model"("id") ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS "maml_data" (
    "id" bigint GENERATED ALWAYS AS IDENTITY NOT NULL UNIQUE,
    "update_lr" varchar(255) NOT NULL,
    "meta_lr" varchar(255) NOT NULL,
    "n_way" bigint NOT NULL,
    "k_spt" bigint NOT NULL,
    "k_qry" bigint NOT NULL,
    "task_num" bigint NOT NULL,
    "update_step" bigint NOT NULL,
    "update_step_test" bigint NOT NULL,
    "model_id" bigint NOT NULL,
    PRIMARY KEY ("id"),
    CONSTRAINT "maml_data_fk9" FOREIGN KEY ("model_id") REFERENCES "model"("id") ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS "epoch" (
    "id" bigint GENERATED ALWAYS AS IDENTITY NOT NULL UNIQUE,
    "epoch" bigint NOT NULL,
    "model_id" bigint NOT NULL,
    PRIMARY KEY ("id"),
    CONSTRAINT "epoch_fk2" FOREIGN KEY ("model_id") REFERENCES "model"("id") ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS "maml_step" (
    "id" bigint GENERATED ALWAYS AS IDENTITY NOT NULL UNIQUE,
    "step" bigint NOT NULL,
    "type" text NOT NULL,
    "epoch_id" bigint NOT NULL,
    PRIMARY KEY ("id"),
    CONSTRAINT "maml_step_fk3" FOREIGN KEY ("epoch_id") REFERENCES "epoch"("id") ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS "maml_update_acc" (
    "id" bigint GENERATED ALWAYS AS IDENTITY NOT NULL UNIQUE,
    "update" bigint NOT NULL,
    "accuracy" varchar(255) NOT NULL,
    "maml_step_id" bigint NOT NULL,
    PRIMARY KEY ("id"),
    CONSTRAINT "maml_update_acc_fk3" FOREIGN KEY ("maml_step_id") REFERENCES "maml_step"("id") ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS "cnn_result" (
    "id" bigint GENERATED ALWAYS AS IDENTITY NOT NULL UNIQUE,
    "value" varchar(255) NOT NULL,
    "type" text NOT NULL,
    "epoch_id" bigint NOT NULL,
    PRIMARY KEY ("id"),
    CONSTRAINT "cnn_result_fk3" FOREIGN KEY ("epoch_id") REFERENCES "epoch"("id") ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS "class" (
    "id" bigint GENERATED ALWAYS AS IDENTITY NOT NULL UNIQUE,
    "name" text NOT NULL,
    "model_id" bigint NOT NULL,
    PRIMARY KEY ("id"),
    CONSTRAINT "class_fk2" FOREIGN KEY ("model_id") REFERENCES "model"("id") ON DELETE CASCADE
);